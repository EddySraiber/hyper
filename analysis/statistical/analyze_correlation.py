#!/usr/bin/env python3
"""
Correlation Analysis Script
"""
import json

# Load correlation results
with open('/home/eddy/Hyper/data/correlation_results.json', 'r') as f:
    correlations = json.load(f)

# Separate historical vs generated samples
historical = [c for c in correlations if c['test_type'] == 'historical']
generated = [c for c in correlations if c['test_type'] == 'generated_sample']

print("=== CORRELATION ANALYSIS RESULTS ===")

# Historical test analysis
if historical:
    historical_correct = sum(1 for c in historical if c['correct_prediction'])
    historical_accuracy = (historical_correct / len(historical)) * 100
    print(f"\n📊 HISTORICAL TEST DATA:")
    print(f"Total Tests: {len(historical)}")
    print(f"Correct Predictions: {historical_correct}")
    print(f"Accuracy: {historical_accuracy:.1f}%")
    
    # Note: Historical data appears to be synthetic test data with perfect correlation
    print("Note: Historical data shows perfect correlation - likely synthetic test data")

# Generated samples analysis
if generated:
    generated_correct = sum(1 for c in generated if c['correct_prediction'])
    generated_accuracy = (generated_correct / len(generated)) * 100
    
    print(f"\n🎲 GENERATED SAMPLE DATA:")
    print(f"Total Tests: {len(generated)}")
    print(f"Correct Predictions: {generated_correct}")
    print(f"Accuracy: {generated_accuracy:.1f}%")
    
    # Analyze by direction prediction
    up_predictions = [c for c in generated if c['predicted_direction'] == 'up']
    down_predictions = [c for c in generated if c['predicted_direction'] == 'down']
    neutral_predictions = [c for c in generated if c['predicted_direction'] == 'neutral']
    
    up_correct = sum(1 for c in up_predictions if c['correct_prediction'])
    down_correct = sum(1 for c in down_predictions if c['correct_prediction'])
    neutral_correct = sum(1 for c in neutral_predictions if c['correct_prediction'])
    
    print(f"\n📈 PREDICTION BREAKDOWN:")
    print(f"UP predictions: {len(up_predictions)}, correct: {up_correct} ({(up_correct/len(up_predictions)*100):.1f}%)")
    print(f"DOWN predictions: {len(down_predictions)}, correct: {down_correct} ({(down_correct/len(down_predictions)*100):.1f}%)")
    print(f"NEUTRAL predictions: {len(neutral_predictions)}, correct: {neutral_correct} ({(neutral_correct/len(neutral_predictions)*100):.1f}%)")
    
    # Analyze confidence vs accuracy
    high_confidence = [c for c in generated if c['confidence_score'] > 0.5]
    low_confidence = [c for c in generated if c['confidence_score'] <= 0.5]
    
    if high_confidence:
        high_conf_correct = sum(1 for c in high_confidence if c['correct_prediction'])
        high_conf_accuracy = (high_conf_correct / len(high_confidence)) * 100
        print(f"\n🎯 HIGH CONFIDENCE (>0.5): {len(high_confidence)} tests, {high_conf_accuracy:.1f}% accuracy")
    
    if low_confidence:
        low_conf_correct = sum(1 for c in low_confidence if c['correct_prediction'])
        low_conf_accuracy = (low_conf_correct / len(low_confidence)) * 100
        print(f"🎯 LOW CONFIDENCE (<=0.5): {len(low_confidence)} tests, {low_conf_accuracy:.1f}% accuracy")

# Overall accuracy
total_tests = len(correlations)
total_correct = sum(1 for c in correlations if c['correct_prediction'])
overall_accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0

print(f"\n🎯 OVERALL CORRELATION ACCURACY: {overall_accuracy:.1f}%")
print(f"Total tests: {total_tests}, Correct: {total_correct}")