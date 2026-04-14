from flask import Flask, render_template, request, redirect, session
import json
import random

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "supersecretkey"

model = SentenceTransformer('all-MiniLM-L6-v2')


def load_questions():
    with open("questions.json", "r") as file:
        return json.load(file)


def evaluate_answer(user_answer, ideal_answer):

    if not user_answer or user_answer.strip() == "":
        return 0

    try:
        user_answer = user_answer.lower().strip()
        ideal_answer = ideal_answer.lower().strip()

        user_emb = model.encode([user_answer])
        ideal_emb = model.encode([ideal_answer])

        similarity = cosine_similarity(user_emb, ideal_emb)[0][0]

        # short answer penalty
        if len(user_answer.split()) < 5:
            similarity *= 0.8

        score = (similarity ** 0.5) * 5

        return round(min(score, 5), 2)

    except:
        return 0


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        session["name"] = request.form["name"]
        session["email"] = request.form["email"]
        return redirect("/portal")
    return render_template("home.html")


@app.route("/portal", methods=["GET", "POST"])
def portal():
    if request.method == "POST":
        session["stream"] = request.form["stream"]
        session["mode"] = request.form["mode"]

        session.pop("questions", None)
        session.pop("q_index", None)
        session.pop("answers", None)

        return redirect("/interview")

    return render_template(
        "portal.html",
        name=session.get("name"),
        email=session.get("email")
    )


@app.route("/interview", methods=["GET", "POST"])
def interview():

    data = load_questions()
    stream = session.get("stream")
    mode = session.get("mode")

    technical_all = data.get(stream, {}).get("technical", [])
    hr = data.get(stream, {}).get("hr", [])

    # ✅ Remove coding in voice mode
    if mode == "voice":
        technical = [q for q in technical_all if q.get("type") != "coding"]
    else:
        technical = technical_all

    # ✅ Create questions once
    if "questions" not in session:

        tech_count = min(5, len(technical))
        tech_q = random.sample(technical, tech_count)

        remaining_needed = 8 - tech_count
        hr_count = min(len(hr), max(3, remaining_needed))
        hr_q = random.sample(hr, hr_count)

        # ✅ Add section tag
        final_questions = []

        for q in tech_q:
            q["section"] = "technical"
            final_questions.append(q)

        for q in hr_q:
            q["section"] = "hr"
            final_questions.append(q)

        final_questions = final_questions[:8]

        session["questions"] = final_questions
        session["q_index"] = 0
        session["answers"] = []

    # ✅ Save answer
    if request.method == "POST":
        answer = request.form.get("answer", "")
        session["answers"].append(answer)
        session["q_index"] += 1

        if session["q_index"] >= len(session["questions"]):
            return redirect("/result")

    # ✅ Current question
    q_index = session.get("q_index", 0)
    current_q = session["questions"][q_index]

    # ✅ Correct heading
    title = "Technical Round" if current_q["section"] == "technical" else "HR Round"

    return render_template(
        "interview.html",
        title=title,
        questions=[current_q],  # ✅ FULL OBJECT PASSED
        number=q_index + 1,
        total=len(session["questions"]),
        remaining=len(session["questions"]) - len(session["answers"])
    )


@app.route("/result")
def result():

    questions = session.get("questions", [])
    answers = session.get("answers", [])

    result_data = []

    for i in range(len(questions)):
        q = questions[i]
        user_answer = answers[i] if i < len(answers) else ""

        score = evaluate_answer(user_answer, q["ideal_answer"])

        result_data.append({
            "question": q["question"],
            "answer": user_answer,
            "score": score
        })

    total_score = round(sum(r["score"] for r in result_data), 2)

    # CALCULATIONS
    tech_scores = [r["score"] for r in result_data[:5]]
    hr_scores = [r["score"] for r in result_data[5:]]

    tech_avg = round(sum(tech_scores)/len(tech_scores), 2) if tech_scores else 0
    hr_avg = round(sum(hr_scores)/len(hr_scores), 2) if hr_scores else 0

    max_score = len(result_data) * 5
    confidence = round((total_score / max_score) * 100, 2)

    # FEEDBACK
    strengths = []
    weaknesses = []
    suggestions = []

    if tech_avg >= 3.5:
        strengths.append("Strong technical knowledge")
    else:
        weaknesses.append("Technical skills need improvement")
        suggestions.append("Practice coding and core concepts")

    if hr_avg >= 3:
        strengths.append("Good communication skills")
    else:
        weaknesses.append("HR answers lack clarity")
        suggestions.append("Improve structured answers")

    if confidence >= 70:
        strengths.append("High overall performance")
    elif confidence >= 40:
        weaknesses.append("Moderate performance")
        suggestions.append("Work on consistency")
    else:
        weaknesses.append("Low performance")
        suggestions.append("Practice more mock interviews")

    if not strengths:
        strengths.append("You are improving steadily, keep practicing!")

    return render_template(
        "result.html",
        results=result_data,
        total=total_score,
        tech_avg=tech_avg,
        hr_avg=hr_avg,
        confidence=confidence,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions
    )


if __name__ == "__main__":
    app.run(debug=True)