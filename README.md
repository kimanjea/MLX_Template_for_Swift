# AVELA AI & XR LAB

A SwiftUI chat client paired with a Python FastAPI + MLX backend for interactive data‑activism tutoring and code hints.

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Configuration](#configuration)
* [Running the Backend](#running-the-backend)
* [Running the iOS/macOS App](#running-the-iosmacos-app)
* [Usage](#usage)
* [Testing](#testing)
* [Contributing](#contributing)
* [License](#license)

## Features

* **SwiftUI Client**: A modern chat UI with suggestion sidebar and real‑time messaging.
* **FastAPI Backend**: Serves `/ask` endpoint to classify, retrieve context, and stream MLX-generated hints.
* **On‑Topic Classifier**: Filters questions to ensure relevance to data‑activism topics.
* **Context Retrieval**: Splits and embeds PDF content for on‑demand hint generation using sentence-transformers.
* **Prompt Caching**: Speeds up inference with MLX prompt cache.

## Prerequisites

* **Xcode 26+** (iOS 16+ or macOS 13+ target)
* **Swift #.#+**
* **Python 3.11+**
* **Python Packages**:

  * `fastapi`
  * `uvicorn`
  * `mlx-lm`
  * `sentence-transformers`
  * `langchain`
  * `transformers`
  * `accelerate`
  * `scikit-learn`
  * `pdfplumber`
  * `datasets`
  * `huggingface-hub`

## Project Structure

Most Important files

```
MLX_Template_for_Swift
|
|--MLX_Researcher_Swift_Final
|         |
|         |--MLX_Researcher_Swift_Final
|                  |
|                  |-- server.py
|                  |-- templatable.py 
|                  |-- ChatViewModel.swift
|                  |-- ContentView.swift
|                  |-- Final_Activity.pdf
|                  |-- Data_activism_classifier
|                  |          | 
|                  |          | --model.safetensors (very important)
|                               
```

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/kimanjea/MLX_Template_for_Swift.git
   cd MLX_Template_for_Swift

   //this is where you open your folder from in xcode 26
   cd MLX_Researcher_Swift_Final

   // this is where your app and server all leave
   cd MLX_Researcher_Swift_Final
   ```

 if you do not have python3.11 installed in your M chip Mac, use these commands in your base terminal

 '''Terminal
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install python@3.11

 
2. **Backend setup**

   ```Terminal
   in 
   cd MLX_Researcher_Swift_Template

   
   
   python3.11 -m pip install mlx_lm
   python3.11 -m pip install langchain
   python3.11 -m pip install sentence-transformers
   python3.11 -m pip install torch
   python3.11 -m pip install pdfplumber
   python3.11 -m pip install datasets
   python3.11 -m pip install scikit-learn pandas 
   python3.11 -m pip install "transformers[torch]" accelerate
   python3.11 -m  pip install 'accelerate>=0.26.0'
   python3.11 -m pip install fastapi
   python3.11 -m  pip install uvicorn
   Python3.11 -m pip install openai

 
   Run your tokens in terminal Line by line on both the main and gpt branch.
   
   export OPENAI_API_KEY=#
   export HUGGINGFACE_HUB_TOKEN=#

//optional but can work without being in specified environment
   python3.11 -m venv venv
   source venv/bin/activate

   ```

3. **iOS/macOS app**

   * Open `SwiftClient/MLX_templateApp.xcodeproj` in Xcode.
   * Ensure the deployment target matches your device or simulator.

## Configuration

* **Backend**:
  * If you have a pre‐built classifier, ensure it lives in `data_activism_classifier/`.
Provided here: if you require a V0 version,

* **Client**:

  * The endpoint URL is configured in `ChatViewModel` as `http://127.0.0.1:8000/ask`.
  * Adjust host/port if your backend differs.

## Running the Backend

Start the FastAPI app with Uvicorn:

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000 
```

## Make sure: NB:
Make sure to delete Mistral.safetensors from the folder before you run the app for a conversation for the first time.
Make sure to delete conversations_log csv from folder so that it can track most recent conversation and turns
Make sure you have a data_activism_classfier within the right directory as mentioned in the instructionsand also it has to include the configs and safetensor files.
Make sure your Xcode app has read and write permission in the sandbox settings
Make sure your xcode app is allowed to make server calls in App Transport Security Settings
Make sure your xcode app has schemes with the tokens about laid out
Make sure your xcode app has MLX and MLXLM frameworks in the Target settings
Make sure you have all installations of python3.11

## Running the iOS/macOS App

1. Build and run in Xcode 26.
2. Interact via the chat UI; type your question or select a suggestion.
3. Messages stream back from the MLX backend.

## Usage

* **Ask a question** about data activism, coding, or visualizations.
* **On‑topic enforcement**: Off‑topic queries will prompt a redirection message.
* **Prompt suggestions**: Predefined starter questions available in the sidebar.

## Testing

* Write unit tests for the classifier and retrieval in `Backend/tests`.
* Use Xcode’s test runner for SwiftUI components.

## Contributing

Contributions are welcome! Please open issues or pull requests against the `develop` branch.

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/YourFeature`.
3. Commit: `git commit -am 'Add new feature'`.
4. Push: `git push origin feature/YourFeature`.
5. Open a PR describing your changes.

## License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
