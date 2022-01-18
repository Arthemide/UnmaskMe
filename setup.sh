#!/bin/bash
mkdir -p ~/.streamlit/
printf "[server]\nheadless = true\nport = %s\nenableCORS = false\n\n" "$PORT" > ~/.streamlit/config.toml