import boto3, pickle, io, json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime, timezone

BUCKET = 'spam-classifier-mlops-yt2'
s3 = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')

# ── Load model ─────────────────────────────────────────────────────
obj = s3.get_object(Bucket=BUCKET, Key='models/naive_bayes_v1.pkl')
model = pickle.load(io.BytesIO(obj['Body'].read()))
print("Model loaded!")

# ── Baseline scores (SMS) ──────────────────────────────────────────
obj = s3.get_object(Bucket=BUCKET, Key='data/train/sms_spam.csv')
sms_df = pd.read_csv(io.BytesIO(obj['Body'].read())).dropna()
sms_df['text'] = sms_df['text'].astype(str)
baseline_scores = model.predict_proba(sms_df['text']).max(axis=1)
baseline_mean = float(baseline_scores.mean())
print(f"Baseline mean confidence: {baseline_mean:.4f}")

# ── Enron batch scores ─────────────────────────────────────────────
obj = s3.get_object(Bucket=BUCKET, Key='data/production_batches/batch_1_enron.csv')
enron_df = pd.read_csv(io.BytesIO(obj['Body'].read())).dropna()
enron_df['text'] = enron_df['text'].astype(str)
new_scores = model.predict_proba(enron_df['text']).max(axis=1)
new_mean = float(new_scores.mean())
print(f"Enron mean confidence: {new_mean:.4f}")

# ── KS Test ────────────────────────────────────────────────────────
stat, p_value = ks_2samp(baseline_scores, new_scores)
drift_detected = bool(p_value < 0.05 or new_mean < 0.80)
print(f"KS statistic: {stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Drift detected: {drift_detected}")

# ── Push to CloudWatch ─────────────────────────────────────────────
cloudwatch.put_metric_data(
    Namespace='SpamClassifier',
    MetricData=[
        {
            'MetricName': 'BatchMeanConfidence',
            'Value': new_mean,
            'Unit': 'None',
            'Timestamp': datetime.now(timezone.utc)
        },
        {
            'MetricName': 'DriftDetected',
            'Value': 1.0 if drift_detected else 0.0,
            'Unit': 'Count',
            'Timestamp': datetime.now(timezone.utc)
        }
    ]
)
print("Metrics pushed to CloudWatch!")