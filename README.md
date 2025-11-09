# Anime Scenario Archetype Generator

This project uses AI to find two thematically similar anime and generate an archetypal plot summary that could apply to both.

## 🚀 Installation

1.  Clone this repository.
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file and add your API key:
    `GEMINI_API_KEY="your_key_here"`

## 🛠️ Setup Steps (Run only once)

To build the database and AI artifacts, run the scripts in order:

1.  `python scripts/1_collect_data.py`
2.  `python scripts/2_filter_franchise.py`
3.  `python scripts/3_create_embeddings.py`

## 🏃 Run the App

Once the setup is complete, run the main application:

```bash
python main_app.py