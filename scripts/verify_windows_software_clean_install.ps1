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
$model = Find-OneFile -Root $installRoot -Filter "Qwen3-0.6B-Q8_0.gguf" -Label "pinned Qwen GGUF model"
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

    Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:$port/v1/profile" -Headers $headers -ContentType "application/json" -Body '{"response_length":"compact","language":"vi","conversation_style":"natural"}' -TimeoutSec 30 | Out-Null

    $chat = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/v1/chat" -Headers $headers -ContentType "application/json" -Body '{"text":"Chỉ trả lời thật ngắn gọn bằng tiếng Việt: xin chào."}' -TimeoutSec 180
    if ([string]::IsNullOrWhiteSpace([string]$chat.reply)) {
        throw "Installed software runtime returned empty chat reply"
    }

    $history = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/history" -Headers $headers -TimeoutSec 30
    if (@($history.messages).Count -lt 2) {
        throw "Installed software runtime did not persist chat history"
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
