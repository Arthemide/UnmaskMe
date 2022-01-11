#!/bin/bash
mkdir -p ~/.streamlit/
printf "[server]\nheadless = true\nport = $PORT\nenableCORS = false\n\n" > ~/.streamlit/config.toml