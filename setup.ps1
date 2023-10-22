# Define the name of the virtual environment
$venvName = ".venv"

# Check if the virtual environment directory already exists
if (!(Test-Path .venv)) {
    # Create a new Python virtual environment
    python -m venv $venvName
}

# Activate the virtual environment
$activateScript = "$venvName\Scripts\Activate"
if (Test-Path $activateScript) {
    . $activateScript
}
else {
    Write-Host "Virtual environment activation script not found. Please activate the environment manually."
}

# Install dependencies from requirements file
$requirementsFile = "requirements.txt"
if (Test-Path $requirementsFile) {
    pip install -r $requirementsFile
}
else {
    Write-Host "Requirements file not found. Please provide a valid file containing dependencies."
}