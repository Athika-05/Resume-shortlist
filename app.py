from flask import Flask, render_template, request
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# Create database
def init_db():
    conn = sqlite3.connect("resumes.db")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            job_description TEXT,
            score REAL,
            status TEXT
        )
    """)

    conn.commit()
    conn.close()


# Extract text from PDF
def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# Calculate resume similarity
def calculate_score(resume_text, job_description):

    if not resume_text.strip():
        return 0

    if not job_description.strip():
        return 0

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        matrix = vectorizer.fit_transform(
            [resume_text, job_description]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return round(score * 100, 2)

    except ValueError:
        return 0


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Process uploaded resume
@app.route("/shortlist", methods=["POST"])
def shortlist():

    job_description = request.form.get(
        "job_description", ""
    ).strip()

    resume = request.files.get("resume")

    if not job_description:
        return "Please enter a job description.", 400

    if not resume or not resume.filename:
        return "Please upload a resume.", 400

    if not resume.filename.lower().endswith(".pdf"):
        return "Only PDF resumes are supported.", 400

    # Generate a safe, unique filename
    import uuid

    filename = str(uuid.uuid4()) + ".pdf"
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"], filename
    )

    resume.save(file_path)

    try:
        resume_text = extract_text(file_path)

        if not resume_text.strip():
            return (
                "No readable text found in the PDF resume.",
                400
            )

        score = calculate_score(
            resume_text,
            job_description
        )

        # Demonstration threshold
        threshold = 50

        if score >= threshold:
            status = "Shortlisted for review"
        else:
            status = "Needs manual review"

        # Store results in SQLite
        conn = sqlite3.connect("resumes.db")

        conn.execute("""
            INSERT INTO candidates
            (filename, job_description, score, status)
            VALUES (?, ?, ?, ?)
        """, (
            resume.filename,
            job_description,
            score,
            status
        ))

        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            filename=resume.filename,
            score=score,
            status=status
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# View previous results
@app.route("/candidates")
def candidates():

    conn = sqlite3.connect("resumes.db")
    conn.row_factory = sqlite3.Row

    records = conn.execute("""
        SELECT * FROM candidates
        ORDER BY score DESC
    """).fetchall()

    conn.close()

    return render_template(
        "result.html",
        records=records
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)