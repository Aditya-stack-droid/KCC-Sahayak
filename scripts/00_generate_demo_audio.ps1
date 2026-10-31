# Synthesizes a short spoken farmer query into demo/sample_query.wav using
# Windows' built-in SAPI text-to-speech, so the ASR (Whisper) stage of the
# pipeline can be exercised on real audio rather than only on typed text.
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$root = Split-Path -Parent $PSScriptRoot
$outDir = Join-Path $root "demo"
if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
$outPath = Join-Path $outDir "sample_query.wav"
$synth.SetOutputToWaveFile($outPath)
$synth.Speak("What is the control measure for aphid infestation in mustard crop")
$synth.Dispose()
Write-Host "Wrote $outPath"
