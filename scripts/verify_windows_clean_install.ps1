
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
        [int]$Attempts = 120
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

$app = Find-OneFile -Root $installRoot -Filter "Nolane.exe" -Label "Nolane.exe"
$runtime = Find-OneFile -Root $installRoot -Filter "nolane-product-runtime.exe" -Label "bundled runtime"
$model = Find-OneFile -Root $installRoot -Filter "factorized-nolane.pt" -Label "factorized model"
$ceremony = Find-OneFile -Root $installRoot -Filter "promotion-ceremony.json" -Label "promotion ceremony"
$tokenizerJson = Find-OneFile -Root $installRoot -Filter "tokenizer.json" -Label "tokenizer.json"
$tokenizerConfig = Find-OneFile -Root $installRoot -Filter "tokenizer_config.json" -Label "tokenizer_config.json"
$releaseManifest = Find-OneFile -Root $installRoot -Filter "release-assets.json" -Label "release-assets.json"

$manifest = Get-Content $releaseManifest.FullName -Raw | ConvertFrom-Json
$modelHash = (Get-FileHash $model.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$runtimeHash = (Get-FileHash $runtime.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
if ($modelHash -ne ([string]$manifest.model_checkpoint_sha256).ToLowerInvariant()) {
    throw "Installed model hash does not match release-assets.json"
}
if ($runtimeHash -ne ([string]$manifest.runtime_executable_sha256).ToLowerInvariant()) {
    throw "Installed runtime hash does not match release-assets.json"
}

$ceremonyPayload = Get-Content $ceremony.FullName -Raw | ConvertFrom-Json
if (([string]$ceremonyPayload.candidate_checkpoint_sha256).ToLowerInvariant() -ne $modelHash) {
    throw "Installed ceremony does not authorize the installed model"
}

$appVersion = [string]$app.VersionInfo.ProductVersion
if (-not $appVersion.StartsWith($ExpectedVersion)) {
    throw "Installed app version mismatch: expected $ExpectedVersion, got $appVersion"
}

$token = ([Guid]::NewGuid().ToString("N") + [Guid]::NewGuid().ToString("N"))
$port = Get-FreePort
$data = Join-Path $env:RUNNER_TEMP "nolane-installed-runtime-data"
if (Test-Path $data) {
    Remove-Item $data -Recurse -Force
}

$runtimeArgs = @(
    "--host", "127.0.0.1",
    "--port", "$port",
    "--data-dir", "$data",
    "--model-bundle", "$($model.Directory.FullName)",
    "--tokenizer", "$($tokenizerJson.Directory.FullName)",
    "--ceremony", "$($ceremony.FullName)",
    "--device", "cpu",
    "--auth-token", "$token"
)

$runtimeProcess = Start-Process -FilePath $runtime.FullName -ArgumentList $runtimeArgs -PassThru -WindowStyle Hidden

$readiness = $null
$power = $null
$chat = $null
$history = $null
try {
    Wait-RuntimeHealth -Port $port -Process $runtimeProcess
    $headers = @{ "X-Nolane-Token" = $token }

    $readiness = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/readiness" -Headers $headers -TimeoutSec 30
    if ($readiness.status -ne "PASS" -or $readiness.critical_failures -ne 0) {
        throw "Installed runtime readiness did not PASS"
    }

    $power = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/v1/power" -Headers $headers -ContentType "application/json" -Body '{"enabled":true}' -TimeoutSec 180
    if ($power.phase -ne "on") {
        throw "Installed runtime failed to reach AI ON"
    }

    Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:$port/v1/profile" -Headers $headers -ContentType "application/json" -Body '{"response_length":"compact","language":"vi"}' -TimeoutSec 30 | Out-Null

    $chat = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$port/v1/chat" -Headers $headers -ContentType "application/json" -Body '{"text":"Chỉ trả lời ngắn gọn: xin chào."}' -TimeoutSec 180
    if ([string]::IsNullOrWhiteSpace([string]$chat.reply)) {
        throw "Installed runtime returned empty chat reply"
    }

    $history = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/history" -Headers $headers -TimeoutSec 30
    if (@($history.messages).Count -lt 2) {
        throw "Installed runtime did not persist chat history"
    }
} finally {
    if (-not $runtimeProcess.HasExited) {
        Stop-Process -Id $runtimeProcess.Id -Force
        $runtimeProcess.WaitForExit()
    }
}

$existingSidecars = @(Get-Process -Name "nolane-product-runtime" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$appProcess = Start-Process -FilePath $app.FullName -PassThru
$spawnedSidecar = $false
try {
    for ($i = 0; $i -lt 80; $i++) {
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
    schema = "NOLANE-V049-WINDOWS-CLEAN-INSTALL-RECEIPT-V1"
    status = "PASS"
    installer_sha256 = (Get-FileHash $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    installed_app_sha256 = (Get-FileHash $app.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    installed_runtime_sha256 = $runtimeHash
    installed_model_sha256 = $modelHash
    promotion_ceremony_sha256 = [string]$ceremonyPayload.ceremony_sha256
    release_manifest_model_sha256 = [string]$manifest.model_checkpoint_sha256
    product_version = $ExpectedVersion
    app_file_version = $appVersion
    readiness_status = [string]$readiness.status
    readiness_critical_failures = [int]$readiness.critical_failures
    ai_phase = [string]$power.phase
    chat_reply_nonempty = -not [string]::IsNullOrWhiteSpace([string]$chat.reply)
    history_messages = @($history.messages).Count
    installed_app_spawned_runtime = $spawnedSidecar
    privacy = [ordered]@{
        contains_chat_text = $false
        contains_auth_token = $false
        contains_user_data_path = $false
    }
}
$receiptObject | ConvertTo-Json -Depth 6 | Set-Content -Path $receipt -Encoding UTF8

Write-Host "Windows clean-install court: PASS"
Write-Host "Receipt: $receipt"
