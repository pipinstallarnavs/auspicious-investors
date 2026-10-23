param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$Project = 'mahurat-research-20261022',
    [string]$Bucket = 'mahurat-research-20261022-results',
    [string]$Destination,
    [string]$TaskName
)
$ErrorActionPreference = 'Stop'
if ($RunId -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}$') { throw 'Invalid run identifier' }
if (-not $Destination) { $Destination = Join-Path (Join-Path (Split-Path $PSScriptRoot -Parent) 'results') $RunId }
$destinationPath = [IO.Path]::GetFullPath($Destination)
$prefix = 'gs://' + $Bucket + '/runs/' + $RunId
$statusText = gcloud storage cat ($prefix + '/status.json') --project=$Project 2>$null
if ($LASTEXITCODE -ne 0) { Write-Output 'The cloud run has not published a status yet.'; exit 0 }
$status = ($statusText -join "`n") | ConvertFrom-Json
if ($status.state -notin @('completed', 'completed_with_errors', 'failed')) {
    Write-Output ('Cloud experiment is still ' + $status.state + '.'); exit 0
}
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
gcloud storage rsync $prefix $destinationPath --recursive --project=$Project
if ($LASTEXITCODE -ne 0) { throw 'Result download failed; the cloud copy is preserved.' }
$checksums = Join-Path $destinationPath 'checksums.json'
if (Test-Path -LiteralPath $checksums) {
    $manifest = Get-Content -LiteralPath $checksums -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($entry in $manifest.PSObject.Properties) {
        $target = [IO.Path]::GetFullPath((Join-Path $destinationPath $entry.Name))
        $destinationRoot = $destinationPath.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        if (-not $target.StartsWith($destinationRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'A checksum path escapes the result folder.'
        }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
            throw ('Checksum mismatch: ' + $entry.Name)
        }
    }
}
@{ run_id = $RunId; state = $status.state; downloaded_at = (Get-Date).ToUniversalTime().ToString('o');
   source = $prefix; checksums_verified = (Test-Path -LiteralPath $checksums) } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destinationPath 'download-complete.json') -Encoding UTF8
if ($TaskName -and $TaskName -match '^MahuratResults-[a-zA-Z0-9_-]+$') {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
Write-Output ('Downloaded ' + $status.state + ' experiment to ' + $destinationPath)
