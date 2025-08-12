# One-shot AI Legal Assistant test
$filePath = "sample_legal_agreement.pdf"
$apiBase = "http://localhost:8000"

Write-Host "1️⃣ Checking API health..."
$health = Invoke-RestMethod -Uri "$apiBase/health" -Method GET
$health | ConvertTo-Json -Depth 3

if ($health.status -ne "healthy") {
    Write-Host "❌ API is not healthy. Exiting."
    exit
}

Write-Host "2️⃣ Uploading PDF..."
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"
$fileBytes = [System.IO.File]::ReadAllBytes($filePath)
$fileContent = [System.Text.Encoding]::GetEncoding("ISO-8859-1").GetString($fileBytes)
$bodyLines = @(
    "--$boundary",
    "Content-Disposition: form-data; name=`"file`"; filename=`"$filePath`"",
    "Content-Type: application/pdf$LF",
    $fileContent,
    "--$boundary--$LF"
) -join $LF

$upload = Invoke-RestMethod -Uri "$apiBase/upload" `
    -Method POST `
    -Body $bodyLines `
    -ContentType "multipart/form-data; boundary=$boundary"

$docId = $upload.document_id
Write-Host "📄 Uploaded. Document ID:" $docId

Write-Host "3️⃣ Waiting for processing..."
do {
    Start-Sleep -Seconds 2
    $status = Invoke-RestMethod -Uri "$apiBase/documents/$docId/status" -Method GET
    Write-Host "   Status:" $status.status
} while ($status.status -eq "processing")

if ($status.status -ne "completed") {
    Write-Host "❌ Processing failed or incomplete."
    exit
}

Write-Host "4️⃣ Asking the bot for a summary..."
$chatBody = @{
    message = "Summarize this document."
    document_id = $docId
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "$apiBase/chat" `
    -Method POST `
    -Body $chatBody `
    -ContentType "application/json"

Write-Host "🤖 Bot Response:"
$response
