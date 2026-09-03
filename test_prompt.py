#!/usr/bin/env python3
"""Test the trained jailbreak detector on your own prompts.

Usage:
  python test_prompt.py "your prompt here"
  python test_prompt.py          # interactive mode, type prompts
"""
import os, sys, re, joblib, numpy as np

# Load saved model + vectorizer (best model = logistic regression)
BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, 'output', 'models', 'logistic_regression.joblib'))
vec   = joblib.load(os.path.join(BASE, 'output', 'models', 'vectorizer.joblib'))

def clean_prompt(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def check(prompt):
    x = vec.transform([clean_prompt(prompt)])
    prob = model.predict_proba(x)[0]          # [P(benign), P(jailbreak)]
    classes = model.classes_                  # ['benign', 'jailbreak']
    idx = int(np.argmax(prob))
    label = classes[idx]
    conf = prob[idx]
    return label, conf, prob

def show(prompt):
    label, conf, prob = check(prompt)
    flag = '🚨 JAILBREAK' if label == 'jailbreak' else '✅ BENIGN'
    print(f'\nPrompt: "{prompt[:120]}{"..." if len(prompt) > 120 else ""}"')
    print(f'  → {flag}  (confidence {conf:.1%})')
    print(f'    benign={prob[0]:.1%}  jailbreak={prob[1]:.1%}')
    return label

if __name__ == '__main__':
    if len(sys.argv) > 1:
        show(sys.argv[1])
    else:
        print('Jailbreak Detector — type prompts, Ctrl+C to exit')
        print('(Try classic jailbreaks like "DAN mode" or "ignore all instructions")')
        while True:
            try:
                p = input('\nYour prompt> ')
            except (EOFError, KeyboardInterrupt):
                print('\nbye'); break
            if p.strip():
                show(p)