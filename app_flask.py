from flask import Flask, render_template, request, redirect, url_for, send_file, Response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from werkzeug.utils import secure_filename
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.0')

# Custom metrics using prometheus_client
upload_counter = Counter(
    'file_uploads_total', 'Total number of file uploads'
)

prediction_counter = Counter(
    'predictions_total', 'Total number of predictions made'
)

@app.route('/metrics')
def metrics_endpoint():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Load model
model_package = joblib.load("power_price_model.pkl")
if isinstance(model_package, dict) and "model" in model_package:
    model = model_package["model"]
    expected_features = model_package.get("features", None)
else:
    model = model_package
    expected_features = None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            upload_counter.inc()  # Increment upload counter
            return redirect(url_for('results', filename=filename))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)

    # Data summary
    summary = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing': df.isnull().sum().sum()
    }

    # Generate plots
    plot_paths = generate_plots(df)

    # Predictions
    predictions, pred_plots, actuals, error_metrics = generate_predictions(df)
    prediction_counter.inc()  # Increment prediction counter

    comparison_rows = None
    if actuals is not None:
        comparison_rows = [
            {'actual': actual, 'predicted': pred, 'diff': abs(pred - actual)}
            for actual, pred in zip(actuals[:20], predictions[:20])
        ]

    return render_template(
        'results.html',
        summary=summary,
        plots=plot_paths,
        predictions=predictions[:20],
        actuals=actuals[:20] if actuals is not None else None,
        comparison_rows=comparison_rows,
        error_metrics=error_metrics,
        pred_plots=pred_plots
    )

def generate_plots(df):
    plots = []
    plot_count = 0

    # Histogram
    if 'MCP (Rs/MWh) *' in df.columns:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df['MCP (Rs/MWh) *'], bins=30, kde=True, ax=ax, color="#1f77b4")
        ax.set_title("Distribution of MCP")
        path = f"static/plot_{plot_count}.png"
        fig.savefig(path)
        plt.close(fig)
        plots.append(path)
        plot_count += 1

    # Time series
    if 'Date' in df.columns and 'MCP (Rs/MWh) *' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df['Date'], df['MCP (Rs/MWh) *'], color="#2ca02c")
        ax.set_title("MCP Over Time")
        path = f"static/plot_{plot_count}.png"
        fig.savefig(path)
        plt.close(fig)
        plots.append(path)

    return plots

def generate_predictions(df):
    actuals = df['MCP (Rs/MWh) *'].tolist() if 'MCP (Rs/MWh) *' in df.columns else None

    feature_df = df.drop(columns=["Date","Time Block","Hour","time",'MCP (Rs/MWh) *'], errors="ignore")
    features = feature_df.select_dtypes(include=["int64","float64"])
    if expected_features is not None:
        features = features.reindex(columns=expected_features, fill_value=0)
    predictions = model.predict(features)

    error_metrics = None
    if actuals is not None:
        import numpy as _np
        actual_array = _np.array(actuals[:len(predictions)], dtype=float)
        pred_array = _np.array(predictions, dtype=float)
        abs_errors = _np.abs(actual_array - pred_array)
        error_metrics = {
            'mae': float(_np.mean(abs_errors)) if len(abs_errors) > 0 else 0.0,
            'max_error': float(_np.max(abs_errors)) if len(abs_errors) > 0 else 0.0,
            'min_error': float(_np.min(abs_errors)) if len(abs_errors) > 0 else 0.0,
            'count': len(abs_errors)
        }

    # Prediction plots
    pred_plots = []
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(predictions, color="#d62728")
    ax.set_title("Predicted Prices")
    path = "static/pred_plot_0.png"
    fig.savefig(path)
    plt.close(fig)
    pred_plots.append(path)

    if actuals is not None:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(actuals[:len(predictions)], label='Actual', color="#1f77b4")
        ax.plot(predictions, label='Predicted', color="#ff7f0e")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        path = "static/pred_plot_1.png"
        fig.savefig(path)
        plt.close(fig)
        pred_plots.append(path)

    return predictions, pred_plots, actuals, error_metrics

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)
