import os
import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic(2).pkl")

# The uploaded artifact is a scikit-learn LogisticRegression model.
# Load only model files that you trust.
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

FEATURES = [
    "Major_Category",
    "Year_of_Study",
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Primary_Use_Case",
    "Prompt_Engineering_Skill",
    "Tool_Diversity",
    "Paid_Subscription",
    "Traditional_Study_Hours",
    "Perceived_AI_Dependency",
    "Institutional_Policy",
    "Anxiety_Level_During_Exams",
    "Post_Semester_GPA",
    "Skill_Retention_Score",
]

# Labels are intentionally shown as model class codes because the pickle
# contains classes [0, 1, 2] but does not include their human-readable meanings.
FIELD_HELP = {
    "Major_Category": "Enter the numeric category code used during model training.",
    "Year_of_Study": "Enter the year value using the same encoding as training data.",
    "Pre_Semester_GPA": "Enter the GPA using the scale used in your dataset.",
    "Weekly_GenAI_Hours": "Approximate weekly hours, using your dataset's units.",
    "Primary_Use_Case": "Enter the numeric category code used during model training.",
    "Prompt_Engineering_Skill": "Enter the score/category code used during training.",
    "Tool_Diversity": "Enter the numeric value used in your dataset.",
    "Paid_Subscription": "Enter the original encoded value (for example, 0/1 only if that is how you trained it).",
    "Traditional_Study_Hours": "Enter study hours using the same units as the training data.",
    "Perceived_AI_Dependency": "Enter the original score/category code used during training.",
    "Institutional_Policy": "Enter the numeric category code used during model training.",
    "Anxiety_Level_During_Exams": "Enter the original score/category code used during training.",
    "Post_Semester_GPA": "Enter the GPA using the scale used in your dataset.",
    "Skill_Retention_Score": "Enter the original score used during training.",
}

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>InsightAI | Student Outcome Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--ink:#10233f;--muted:#65758b;--navy:#102b4e;--blue:#2563eb;--cyan:#38bdf8;--paper:#f4f8fc;--line:#e4ebf4;--white:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif}
body:before{content:"";position:fixed;inset:0;z-index:-1;background:radial-gradient(circle at 85% 8%,#d9f2ff 0,transparent 28%),radial-gradient(circle at 4% 38%,#e4eaff 0,transparent 27%)}
nav{height:76px;display:flex;align-items:center;justify-content:space-between;padding:0 max(5vw,24px);background:#ffffffd9;backdrop-filter:blur(14px);border-bottom:1px solid #e8eef6;position:sticky;top:0;z-index:5}
.brand{display:flex;gap:11px;align-items:center;font-family:Manrope;font-weight:800;font-size:19px;letter-spacing:-.5px}.logo{width:38px;height:38px;border-radius:13px;background:linear-gradient(135deg,#2563eb,#38bdf8);display:grid;place-items:center;color:white;font-size:20px;box-shadow:0 8px 20px #2563eb35}
.navtag{font-size:12px;font-weight:700;color:#2762b6;background:#eaf3ff;padding:9px 13px;border-radius:99px;letter-spacing:.5px}
.hero{max-width:1180px;margin:58px auto 30px;padding:0 24px;display:grid;grid-template-columns:1.25fr .75fr;gap:35px;align-items:center}
.eyebrow{color:#2563eb;text-transform:uppercase;font-size:11px;font-weight:800;letter-spacing:2px}.hero h1{font-family:Manrope;font-size:clamp(36px,5vw,58px);line-height:1.07;letter-spacing:-2.6px;margin:14px 0}.hero h1 span{background:linear-gradient(90deg,#2563eb,#0ea5e9);-webkit-background-clip:text;color:transparent}.hero p{color:var(--muted);font-size:16px;line-height:1.8;max-width:590px}
.hero-card{background:linear-gradient(145deg,#102b4e,#174b80);border-radius:26px;padding:27px;color:white;box-shadow:0 22px 55px #173e6c25;position:relative;overflow:hidden;animation:float 5s ease-in-out infinite}
.hero-card:after{content:"";position:absolute;width:220px;height:220px;border:1px solid #ffffff20;border-radius:50%;right:-75px;top:-95px;box-shadow:0 0 0 28px #ffffff08,0 0 0 58px #ffffff06}
.spark{font-size:28px}.hero-card h3{font-family:Manrope;font-size:21px;margin:15px 0 8px}.hero-card p{color:#c8d9ee;font-size:13px;line-height:1.7}.metric{display:flex;gap:10px;margin-top:22px}.metric div{background:#ffffff12;border:1px solid #ffffff1c;border-radius:13px;padding:12px 14px;flex:1}.metric b{display:block;font-size:18px}.metric small{color:#c8d9ee}
.wrap{max-width:1180px;margin:0 auto 80px;padding:0 24px}.panel{background:white;border:1px solid var(--line);border-radius:24px;box-shadow:0 18px 55px #1d426b0c;overflow:hidden}
.panel-head{padding:27px 30px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:16px;align-items:center}.panel-head h2{font-family:Manrope;margin:0;font-size:22px}.panel-head p{margin:6px 0 0;color:var(--muted);font-size:13px}.step{font-size:11px;font-weight:800;color:#2563eb;background:#eff6ff;padding:9px 13px;border-radius:99px;white-space:nowrap}
form{padding:28px 30px 32px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:19px}.field label{font-size:13px;font-weight:700;display:block;margin-bottom:8px}.field input{width:100%;height:48px;border:1px solid #dce5ef;border-radius:11px;padding:0 13px;font:500 14px 'DM Sans';color:var(--ink);background:#fbfdff;outline:none;transition:.2s}.field input:focus{border-color:#4795ef;box-shadow:0 0 0 4px #3b82f61a;background:white}.hint{display:block;color:#8795a8;font-size:11px;line-height:1.5;margin-top:6px}
.actions{display:flex;align-items:center;gap:16px;margin-top:28px;flex-wrap:wrap}.btn{border:0;border-radius:12px;padding:15px 23px;color:white;background:linear-gradient(100deg,#2563eb,#0ea5e9);font:700 14px 'DM Sans';cursor:pointer;box-shadow:0 9px 22px #2563eb30;transition:transform .2s,box-shadow .2s}.btn:hover{transform:translateY(-2px);box-shadow:0 13px 26px #2563eb40}.note{color:#8795a8;font-size:11px;line-height:1.5}
.result{margin:0 30px 30px;padding:22px 24px;border-radius:17px;background:linear-gradient(110deg,#eff6ff,#f0faff);border:1px solid #dbeafe;animation:rise .5s ease both}.result .kicker{font-size:10px;font-weight:800;color:#2563eb;letter-spacing:1.5px;text-transform:uppercase}.result h3{font:800 25px Manrope;margin:8px 0 5px}.result p{color:#5d718b;font-size:13px;margin:0;line-height:1.6}.error{margin:0 30px 25px;padding:15px;border-radius:12px;background:#fff1f2;color:#be123c;font-size:13px}
footer{text-align:center;color:#8997a9;font-size:12px;padding:0 20px 35px}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@media(max-width:760px){.hero{grid-template-columns:1fr;margin-top:38px}.hero-card{display:none}.grid{grid-template-columns:1fr}.panel-head{align-items:flex-start;flex-direction:column}form{padding:22px 20px}.panel-head{padding:23px 20px}.result,.error{margin-left:20px;margin-right:20px}.hero h1{letter-spacing:-1.5px}}
</style>
</head>
<body>
<nav><div class="brand"><div class="logo">✦</div>InsightAI</div><div class="navtag">MACHINE LEARNING · LIVE DEMO</div></nav>
<section class="hero">
 <div><div class="eyebrow">A smarter way to explore outcomes</div><h1>Turn student data into <span>clear insights.</span></h1>
 <p>Explore a machine-learning prediction using your student and study-related inputs. Enter values in the same format and encoding used to train the model.</p></div>
 <div class="hero-card"><div class="spark">✧</div><h3>Student Outcome Predictor</h3><p>Powered by a Logistic Regression model. A clean, simple interface for exploring model predictions.</p><div class="metric"><div><b>14</b><small>Input features</small></div><div><b>3</b><small>Model classes</small></div></div></div>
</section>
<main class="wrap"><section class="panel">
 <div class="panel-head"><div><h2>Prediction workspace</h2><p>Complete all 14 fields to generate a model prediction.</p></div><div class="step">01 / INPUT DATA</div></div>
 <form method="POST">
  <div class="grid">
  {% for f in features %}
   <div class="field"><label for="{{f}}">{{ f.replace('_',' ') }}</label><input id="{{f}}" name="{{f}}" type="number" step="any" required value="{{ values.get(f,'') }}" placeholder="Enter numeric value"><span class="hint">{{ help_text[f] }}</span></div>
  {% endfor %}
  </div>
  <div class="actions"><button class="btn" type="submit">Generate prediction&nbsp; →</button><span class="note">Use the original dataset's numeric category codes and scales.<br>Predictions are exploratory, not academic or personal assessments.</span></div>
 </form>
 {% if prediction is not none %}
 <div class="result"><div class="kicker">Prediction generated</div><h3>Predicted class: {{ prediction }}</h3><p>The model returned class code <b>{{ prediction }}</b>. The uploaded model contains class codes 0, 1 and 2, but no human-readable class labels. Map this code to your target label from the original training dataset.</p></div>
 {% endif %}
 {% if error %}<div class="error">{{error}}</div>{% endif %}
</section></main>
<footer>InsightAI · Logistic Regression Demo · Built with Flask & scikit-learn</footer>
</body></html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    values = {}
    if request.method == "POST":
        try:
            row = []
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()
                values[feature] = raw
                if raw == "":
                    raise ValueError(f"Please enter a value for {feature.replace('_', ' ')}.")
                row.append(float(raw))
            if len(row) != int(model.n_features_in_):
                raise ValueError("The number of inputs does not match the model's expected feature count.")
            prediction = int(model.predict(np.asarray([row], dtype=float))[0])
        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "The model could not process these values. Check that every value uses the same encoding and scale as the training data."

    return render_template_string(
        HTML, features=FEATURES, help_text=FIELD_HELP,
        prediction=prediction, error=error, values=values
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
