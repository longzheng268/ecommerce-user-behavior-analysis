#!/usr/bin/env python3
"""E-Commerce Analysis Dashboard - Flask App Entry Point."""
import os
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template
import config
from api.routes import api

app = Flask(__name__)
app.register_blueprint(api)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    print("🚀 Starting E-Commerce Analysis Dashboard...")
    print(f"   http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
