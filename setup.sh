mkdir -p ~/.streamlit/

echo "[general]
email = \"your-email@domain.com\"
" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
enableCORS = false
port = 8510
" > ~/.streamlit/config.toml
