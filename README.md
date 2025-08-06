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
|                  |-- Data_activism_activity.pdf
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

2. **Backend setup**

   ```bash
   cd MLX_Swift_Template

   python3.11 -m venv venv
   source venv/bin/activate
   python3.11 -m pip install mlx_lm
   python3.11 -m pip install langchain
   python3.11 -m pip install sentence-transformers
   python3.11 -m pip install torch
   python3.11 -m pip install pdfplumber
   python3.11 -m pip install datasets
   python3.11 -m pip install scikit-learn pandas 
   python3.11 -m pip install "transformers[torch]" accelerate
   python3.11 -m  pip install 'accelerate>=0.26.0'

   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **iOS/macOS app**

   * Open `SwiftClient/MLX_templateApp.xcodeproj` in Xcode.
   * Ensure the deployment target matches your device or simulator.

## Configuration

* **Backend**:

  * Place your PDF(s) into `Backend/` (default: `Data_Activisim_Piechart_Activity.pdf`).
  * Provide your Hugging Face token via `huggingface-cli login` or env var.
  * If you have a pre‐built classifier, ensure it lives in `data_activism_classifier/`.

* **Client**:

  * The endpoint URL is configured in `ChatViewModel` as `http://127.0.0.1:8000/ask`.
  * Adjust host/port if your backend differs.

## Running the Backend

Start the FastAPI app with Uvicorn:

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000 
```

* **`/ask`**: Accepts `POST` JSON `{ "question": "..." }`, returns `{ "answer": "..." }`.

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
