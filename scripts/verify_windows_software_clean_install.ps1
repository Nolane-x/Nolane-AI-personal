param(
    [Parameter(Mandatory = $true)]
    [string]$Installer,
    [Parameter(Mandatory = $true)]
    [string]$InstallDir,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedVersion,
    [Parameter(Mandatory = $true)]
    [string]$ReceiptPath
)

$ErrorActionPreference = "Stop"

function Get-FreePort {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        return ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    } finally {
        $listener.Stop()
    }
}

function Wait-RuntimeHealth {
    param(
        [int]$Port,
        [System.Diagnostics.Process]$Process,
        [int]$Attempts = 160
    )
    for ($i = 0; $i -lt $Attempts; $i++) {
        Start-Sleep -Milliseconds 250
        if ($Process.HasExited) {
            throw "Installed Nolane runtime exited during startup"
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/health" -TimeoutSec 1
            if ($health.status -eq "ok") {
                return
            }
        } catch {}
    }
    throw "Installed Nolane runtime did not become healthy"
}

function Find-OneFile {
    param(
        [string]$Root,
        [string]$Filter,
        [string]$Label
    )
    $matches = @(Get-ChildItem -Path $Root -Filter $Filter -Recurse -File -ErrorAction Stop)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one $Label under $Root; found $($matches.Count)"
    }
    return $matches[0]
}

$installerPath = (Resolve-Path $Installer).Path
$installRoot = [System.IO.Path]::GetFullPath($InstallDir)
$receipt = [System.IO.Path]::GetFullPath($ReceiptPath)

if (Test-Path $installRoot) {
    Remove-Item -Path $installRoot -Recurse -Force
}
New-Item -ItemType Directory -Force (Split-Path $receipt -Parent) | Out-Null

$installerProcess = Start-Process -FilePath $installerPath -ArgumentList @("/S", "/D=$installRoot") -PassThru -Wait
if ($installerProcess.ExitCode -ne 0) {
    throw "NSIS clean install failed with exit code $($installerProcess.ExitCode)"
}

$app = Find-OneFile -Root $installRoot -Filter "nolane-product-client.exe" -Label "nolane-product-client.exe"
$model = Find-OneFile -Root $installRoot -Filter "Qwen_Qwen3.5-2B-Q4_K_M.gguf" -Label "pinned Qwen GGUF model"
$manifest = Find-OneFile -Root $installRoot -Filter "software-release.json" -Label "software release manifest"
$runtime = Find-OneFile -Root $installRoot -Filter "nolane-product-runtime.exe" -Label "Nolane product runtime"
$llamaServer = Find-OneFile -Root $installRoot -Filter "llama-server.exe" -Label "llama.cpp server"
$uninstaller = Find-OneFile -Root $installRoot -Filter "uninstall.exe" -Label "NSIS uninstaller"

$manifestPayload = Get-Content $manifest.FullName -Raw | ConvertFrom-Json
if ([string]$manifestPayload.schema -ne "NOLANE-V100-WINDOWS-SOFTWARE-RELEASE-V1") {
    throw "Software release manifest schema mismatch"
}
if ([string]$manifestPayload.authority -ne "CI_SOFTWARE_RELEASE_PINNED_UPSTREAM_RUNTIME") {
    throw "Software release manifest authority mismatch"
}

$modelHash = (Get-FileHash $model.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$runtimeHash = (Get-FileHash $runtime.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$llamaHash = (Get-FileHash $llamaServer.FullName -Algorithm SHA256).Hash.ToLowerInvariant()

if ($modelHash -ne ([string]$manifestPayload.model_sha256).ToLowerInvariant()) {
    throw "Installed GGUF model hash does not match manifest"
}
if ($runtimeHash -ne ([string]$manifestPayload.runtime_executable_sha256).ToLowerInvariant()) {
    throw "Installed Nolane runtime hash does not match manifest"
}
if ($llamaHash -ne ([string]$manifestPayload.llama_server_sha256).ToLowerInvariant()) {
    throw "Installed llama-server hash does not match manifest"
}
if (-not [bool]$manifestPayload.windows_one_click_prerequisites_bundled) {
    throw "Software manifest does not assert bundled prerequisites"
}
if ([bool]$manifestPayload.release_claims.l36_certified) {
    throw "Software release must not claim L36 certification"
}

$appVersion = [string]$app.VersionInfo.ProductVersion
if (-not $appVersion.StartsWith($ExpectedVersion)) {
    throw "Installed app version mismatch: expected $ExpectedVersion, got $appVersion"
}

$token = ([Guid]::NewGuid().ToString("N") + [Guid]::NewGuid().ToString("N"))
$port = Get-FreePort
$data = Join-Path $env:RUNNER_TEMP "nolane-software-clean-install-data"
if (Test-Path $data) {
    Remove-Item $data -Recurse -Force
}

$runtimeArgs = @(
    "--host", "127.0.0.1",
    "--port", "$port",
    "--data-dir", "$data",
    "--software-model", "$($model.FullName)",
    "--llama-server", "$($llamaServer.FullName)",
    "--software-manifest", "$($manifest.FullName)",
    "--device", "cpu",
    "--auth-token", "$token"
)

$existingLlama = @(Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$runtimeProcess = Start-Process -FilePath $runtime.FullName -ArgumentList $runtimeArgs -PassThru -WindowStyle Hidden

$readiness = $null
$power = $null
$chat = $null
$history = $null
$qualityPassed = 0
$qualityTotal = 0
$spawnedLlama = $false
try {
    Wait-RuntimeHealth -Port $port -Process $runtimeProcess
    $headers = @{ "X-Nolane-Token" = $token }

    $readiness = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/readiness" -Headers $headers -TimeoutSec 30
    if ($readiness.status -ne "PASS" -or $readiness.critical_failures -ne 0) {
        throw "Installed software runtime readiness did not PASS"
    }

    $power = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/v1/power" -Headers $headers -ContentType "application/json" -Body '{"enabled":true}' -TimeoutSec 180
    if ($power.phase -ne "on") {
        throw "Installed software runtime failed to reach AI ON: $($power.error)"
    }
    if ([string]$power.runtime_channel -ne "software-v1-gguf") {
        throw "Installed runtime did not select software-v1-gguf"
    }

    $newLlama = @(Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Where-Object { $existingLlama -notcontains $_.Id })
    $spawnedLlama = $newLlama.Count -gt 0
    if (-not $spawnedLlama) {
        throw "Power-on did not start bundled llama-server"
    }

    $profileBody = @{
        preferred_name = "Huy"
        assistant_name = "Mây"
        response_length = "balanced"
        language = "vi"
        conversation_style = "natural"
    } | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:$port/v1/profile" -Headers $headers -ContentType "application/json" -Body $profileBody -TimeoutSec 30 | Out-Null

    function Assert-Quality {
        param(
            [string]$Name,
            [bool]$Condition,
            [string]$Detail
        )
        $script:qualityTotal += 1
        if (-not $Condition) {
            throw "Conversation quality gate '$Name' failed: $Detail"
        }
        $script:qualityPassed += 1
        Write-Host "[conversation-court] PASS: $Name"
    }

    function Invoke-QualityChat {
        param([string]$Text)
        $body = @{ text = $Text } | ConvertTo-Json -Compress
        $result = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/v1/chat" -Headers $headers -ContentType "application/json" -Body $body -TimeoutSec 180
        $reply = [string]$result.reply
        Write-Host ""
        Write-Host "[conversation-court] USER: $Text"
        Write-Host "[conversation-court] NOLANE: $reply"
        if ([string]::IsNullOrWhiteSpace($reply)) {
            throw "Conversation court received an empty reply"
        }
        if (
            $reply -match '(?i)preferred_name=' -or
            $reply -match '(?i)assistant_name=' -or
            $reply -match '(?i)runtime state:' -or
            $reply -match '(?i)requested_intent='
        ) {
            throw "Conversation court detected leaked runtime context: $reply"
        }
        return $result
    }

    $chat = Invoke-QualityChat "Chỉ trả lời đúng hai từ: Xin chào"
    Assert-Quality "vietnamese-instruction" ([string]$chat.reply -match '(?i)^\s*xin\s+chào[.!]?\s*$') ([string]$chat.reply)

    $fact = Invoke-QualityChat "Thủ đô của nước Pháp là gì? Trả lời trực tiếp."
    Assert-Quality "basic-fact-paris" ([string]$fact.reply -match '(?i)paris') ([string]$fact.reply)

    $math = Invoke-QualityChat "Lan có 3 quả táo, được cho thêm 4 quả. Lan có tất cả bao nhiêu quả táo?"
    Assert-Quality "simple-reasoning" ([string]$math.reply -match '(^|[^0-9])7([^0-9]|$)') ([string]$math.reply)

    $identity = Invoke-QualityChat "Tên bạn là gì, và tên tôi là gì? Trả lời rõ cả hai."
    $identityText = ([string]$identity.reply).ToLowerInvariant()
    Assert-Quality "identity-separation" (($identityText -match 'mây') -and ($identityText -match 'huy')) ([string]$identity.reply)

    $capability = Invoke-QualityChat "Bạn làm được gì? Hãy trả lời cụ thể những việc bạn có thể giúp tôi."
    $capabilityText = [string]$capability.reply
    $capabilityUseful = $capabilityText.Length -ge 45 -and
        $capabilityText -notmatch '(?i)^\s*bạn làm được gì\??\s*$' -and
        $capabilityText -notmatch '(?i)bạn đang cần gì' -and
        $capabilityText -match '(?i)(giúp|giải thích|phân tích|viết|dịch|tóm tắt|lập kế hoạch|ý tưởng|học|trò chuyện)'
    Assert-Quality "capability-answer" $capabilityUseful $capabilityText

    $remember = Invoke-QualityChat "Trong cuộc trò chuyện này, hãy nhớ mã thử nghiệm NOLANE-2719. Chỉ xác nhận ngắn gọn."
    Assert-Quality "context-write" (-not [string]::IsNullOrWhiteSpace([string]$remember.reply)) ([string]$remember.reply)

    $recall = Invoke-QualityChat "Mã thử nghiệm tôi vừa nói là gì?"
    Assert-Quality "multi-turn-recall" ([string]$recall.reply -match '(?i)NOLANE-2719') ([string]$recall.reply)

    $correction = Invoke-QualityChat "Berlin là thủ đô của Pháp đúng không? Nếu sai hãy sửa lại."
    $correctionText = [string]$correction.reply
    Assert-Quality "correction-repair" (($correctionText -match '(?i)paris') -and ($correctionText -notmatch '(?i)berlin\s+là\s+thủ\s+đô\s+(của\s+)?pháp')) $correctionText

    $reasoning = Invoke-QualityChat "An cao hơn Bình, Bình cao hơn Cường. Ai cao nhất?"
    Assert-Quality "relational-reasoning" ([string]$reasoning.reply -match '(?i)\ban\b') ([string]$reasoning.reply)

    $preference = Invoke-QualityChat "Tôi thích cà phê hơn trà. Hãy nhớ điều này trong cuộc trò chuyện."
    Assert-Quality "preference-write" (-not [string]::IsNullOrWhiteSpace([string]$preference.reply)) ([string]$preference.reply)

    $preferenceRecall = Invoke-QualityChat "Tôi vừa nói mình thích đồ uống nào hơn?"
    Assert-Quality "preference-recall" ([string]$preferenceRecall.reply -match '(?i)cà\s*phê') ([string]$preferenceRecall.reply)

    $toolHonesty = Invoke-QualityChat "Trong phiên bản đang chạy này, bạn có thể tự mở trình duyệt và gửi email cho tôi ngay bây giờ không?"
    $toolText = [string]$toolHonesty.reply
    Assert-Quality "capability-honesty" ($toolText -match '(?i)(không|chưa|không thể|không có)') $toolText

    $englishPrompt = Invoke-QualityChat "What are three useful things you can help me with?"
    $englishText = [string]$englishPrompt.reply
    Assert-Quality "configured-vietnamese-lock" ($englishText -match '(?i)(bạn|mình|tôi|giúp|có thể)') $englishText

    $history = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/history" -Headers $headers -TimeoutSec 30
    if (@($history.messages).Count -lt 26) {
        throw "Installed software runtime did not persist the full conversation court history"
    }
} finally {
    if (-not $runtimeProcess.HasExited) {
        Stop-Process -Id $runtimeProcess.Id -Force
        $runtimeProcess.WaitForExit()
    }
    Get-Process -Name "llama-server" -ErrorAction SilentlyContinue |
        Where-Object { $existingLlama -notcontains $_.Id } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

$existingSidecars = @(Get-Process -Name "nolane-product-runtime" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$appProcess = Start-Process -FilePath $app.FullName -PassThru
$spawnedSidecar = $false
try {
    for ($i = 0; $i -lt 100; $i++) {
        Start-Sleep -Milliseconds 250
        if ($appProcess.HasExited) {
            throw "Installed Nolane app exited during launch"
        }
        $sidecars = @(Get-Process -Name "nolane-product-runtime" -ErrorAction SilentlyContinue)
        $newSidecars = @($sidecars | Where-Object { $existingSidecars -notcontains $_.Id })
        if ($newSidecars.Count -gt 0) {
            $spawnedSidecar = $true
            break
        }
    }
    if (-not $spawnedSidecar) {
        throw "Installed Nolane app did not start its bundled runtime"
    }
} finally {
    if (-not $appProcess.HasExited) {
        Stop-Process -Id $appProcess.Id -Force
    }
    Get-Process -Name "nolane-product-runtime" -ErrorAction SilentlyContinue |
        Where-Object { $existingSidecars -notcontains $_.Id } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

$receiptObject = [ordered]@{
    schema = "NOLANE-V100-WINDOWS-SOFTWARE-CLEAN-INSTALL-RECEIPT-V1"
    authority = "CI_SOFTWARE_RELEASE_PINNED_UPSTREAM_RUNTIME"
    status = "PASS"
    installer_sha256 = (Get-FileHash $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    installed_app_sha256 = (Get-FileHash $app.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    installed_runtime_sha256 = $runtimeHash
    installed_model_sha256 = $modelHash
    installed_llama_server_sha256 = $llamaHash
    product_version = $ExpectedVersion
    app_file_version = $appVersion
    runtime_channel = [string]$power.runtime_channel
    readiness_status = [string]$readiness.status
    readiness_critical_failures = [int]$readiness.critical_failures
    ai_phase = [string]$power.phase
    chat_reply_nonempty = -not [string]::IsNullOrWhiteSpace([string]$chat.reply)
    conversation_quality_passed = $qualityPassed
    conversation_quality_total = $qualityTotal
    conversation_quality_status = if ($qualityPassed -eq $qualityTotal -and $qualityTotal -ge 13) { "PASS" } else { "FAIL" }
    history_messages = @($history.messages).Count
    bundled_llama_server_spawned = $spawnedLlama
    installed_app_spawned_runtime = $spawnedSidecar
    uninstaller_present = $uninstaller.Exists
    model_repo = [string]$manifestPayload.model_repo
    model_revision = [string]$manifestPayload.model_revision
    llama_cpp_tag = [string]$manifestPayload.llama_cpp_tag
    privacy = [ordered]@{
        contains_chat_text = $false
        contains_auth_token = $false
        contains_user_data_path = $false
    }
}
$receiptObject | ConvertTo-Json -Depth 6 | Set-Content -Path $receipt -Encoding UTF8

Write-Host "Windows software clean-install court: PASS"
Write-Host "Receipt: $receipt"
