# Import required libraries
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import spacy
import numpy as np

# Download NLTK data (if not already downloaded)
import nltk.data
nltk.download('punkt')
nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet', download_dir='D:/BTech IT/Advenced Module/Python/chatbot/nltk_data/')  # Add this line
nltk.data.path.append(r"C:\Users\Iraguha\AppData\Roaming\nltk_data")
nltk.data.path.append(r"D:\BTech IT\Advenced Module\Python\chatbot\nltk_data")
#nltk.download('wordnet', download_dir='D:/BTech IT/Advenced Module/Python/chatbot/nltk_data/')  # Add this line


# If you are unsure about what you need, you can launch the NLTK downloader GUI:
nltk.download()

# Initialize Flask app
app = Flask(__name__)

# Sample question-answer dictionary
qa_dict = {
    "What are the admission requirements?": "Admission requirements vary by program. Please check the admissions page on our website.",
    "When is the application deadline?": "Application deadlines depend on the intake. Please refer to our academic calendar.",
    "Is there an application fee?": "Yes, the application fee is 5000Frw for undergraduate programs and 6000Frw for postgraduate programs.",
    "Do you offer scholarships?": "Yes, we offer need-based and merit-based scholarships.",
    "What documents are needed for admission?": "Documents include transcripts, a statement of purpose, recommendation letters, and proof of identification.",
    "How can I track my application status?": "Log in to your applicant portal to check the status.",
    "Can I apply for multiple programs?": "Yes, you can apply for up to three programs simultaneously.",
    "Do you offer online courses?": "Yes, we offer a range of online courses and degrees.",
    "What is the duration of undergraduate programs?": "Most undergraduate programs are four years.",
    "Can I transfer credits from another university?": "Yes, we accept transfer credits after an evaluation process.",
    "What are the admission requirements for undergraduate programs?": "Academic transcripts, proof of identification, and completed application form.",
    "What are the admission requirements for graduate programs?": "Bachelor's degree, transcripts, recommendation letters, and a statement of purpose.",
    "What is the application deadline?": "March for Fall intake, October for Spring intake.",
    "How can I apply online?": "Visit the university's website and complete the online application form.",
    "What is the admission process for international students?": "Submit additional documents like visas and English proficiency test results.",
    "Is there an application fee?": "Yes, the application fee is $50.",
    "Can I apply to multiple programs simultaneously?": "Yes, you can apply to multiple programs.",
    "What documents are required for admission?": "Transcripts, identification, application form, and test scores (if applicable).",
    "Are there any entrance exams required?": "Some programs require entrance exams, like SAT or GRE.",
    "How do I know if my application was received?": "You will receive an email confirmation.",
    "How long does it take to process an application?": "Processing takes 2-4 weeks.",
    "Are there scholarships available for new students?": "Yes, merit-based and need-based scholarships are available.",
    "What are the eligibility criteria for scholarships?": "High academic performance and extracurricular achievements.",
    "How can I check the status of my application?": "Log into the application portal for updates.",
    "Can I defer my admission?": "Yes, deferral is allowed for up to one academic year.",
    "What is the refund policy for the application fee?": "Application fees are non-refundable.",
    "Are there any special requirements for transfer students?": "Yes, you need transcripts from the previous institution and a transfer request.",
    "Can I apply if my previous education is from a different country?": "Yes, but equivalency evaluation might be required.",
    "Is there a minimum GPA requirement for admission?": "Undergraduate programs typically require a minimum GPA of 2.5.",
    "How do I submit my academic transcripts?": "Upload transcripts during the application process.",
    "Are recommendation letters required?": "Yes, at least two letters are required.",
    "What is the process for applying for financial aid?": "Fill out a financial aid form and provide supporting documents.",
    "When will I receive my admission decision?": "Admission decisions are sent out within two weeks of processing.",
    "Are there open house or campus tour events for prospective students?": "Yes, open houses are held annually in spring and fall.",
    "Can I reapply if my application is denied?": "Yes, you can reapply after six months.",
    "What undergraduate courses are offered?": "Engineering, Business, Arts, IT, and Health Sciences.",
    "What graduate programs are available?": "Master’s programs in Business, Science, and Engineering.",
    "Are there any diploma or certificate courses?": "Yes, diploma and certificate courses are available.",
    "Are there online courses offered?": "Yes, online courses in various fields are offered.",
    "Can I take part-time courses?": "Yes, many courses have part-time options.",
    "What are the most popular courses?": "Business, Computer Science, and Engineering are the most popular.",
    "What courses are available in engineering?": "Mechanical, Civil, Electrical, and Computer Engineering courses.",
    "Do you offer medical-related programs?": "Yes, we offer nursing and healthcare programs.",
    "Are there any business courses?": "Business Management, Accounting, and Marketing courses.",
    "Is there a course for computer science or IT?": "Yes, multiple programs in Computer Science and IT are available.",
    "What are the humanities and arts programs offered?": "Programs like Literature, History, and Philosophy.",
    "Are there any language courses available?": "French, Spanish, Mandarin, and English courses.",
    "Do you have short-term training programs?": "Yes, short-term courses in technology and business.",
    "Can I change my course after enrollment?": "Yes, but approval from the academic office is required.",
    "What are the prerequisites for enrolling in specific courses?": "Basic academic qualifications and prerequisite courses.",
    "Do you offer courses for working professionals?": "Yes, evening and weekend programs are designed for professionals.",
    "Are there any internship opportunities associated with the courses?": "Many programs include internship opportunities.",
    "How many credits are required to complete a degree?": "A minimum of 120 credits is needed for undergraduate degrees.",
    "What are the research opportunities for graduate students?": "Graduate students can participate in funded research projects.",
    "Are there dual-degree programs available?": "Yes, dual-degree options are available in select fields.",
    "What courses are available in the evening?": "Evening classes are available in Business and IT.",
    "Are there any free or subsidized courses?": "Yes, some courses are free or heavily subsidized.",
    "How can I get the course syllabus?": "Download the syllabus from the course catalog.",
    "Can I transfer credits from other universities?": "Yes, with approval from the academic office.",
    "Are there courses taught in other languages besides English?": "Yes, select courses are taught in French and Spanish.",
    "What undergraduate programs do you offer?": "We offer programs in engineering, arts, sciences, business, and IT.",
    "What postgraduate programs are available?": "We offer master's and doctoral programs in various disciplines.",
    "Do you offer double majors?": "Yes, students can pursue double majors in approved combinations.",
    "What is the class size for lectures?": "Class sizes vary but typically range from 30 to 50 students.",
    "What language are courses taught in?": "Most courses are taught in English.",
    "Where is the university located?": "The campus is located in [City Name].",
    "How can I get to the campus?": "By public transport, car, or university shuttle.",
    "Are there parking facilities on campus?": "Yes, ample parking is available for students and visitors.",
    "Does the university provide shuttle services?": "Yes, shuttle services run between campus and key locations.",
    "Are there dormitories or hostels available?": "Yes, dormitories are available for local and international students.",
    "What dining options are available on campus?": "Cafeterias, food courts, and vending machines are on campus.",
    "Is there a gym or fitness center?": "Yes, there is a fully equipped gym and fitness center.",
    "Does the campus have a library?": "Yes, the library has extensive collections and study spaces.",
    "What are the opening hours of the library?": "The library is open from 8 AM to 10 PM.",
    "Are there study rooms or lounges on campus?": "Yes, lounges and study rooms are provided in all buildings.",
    "Is there Wi-Fi available on campus?": "Yes, free Wi-Fi is available across the campus.",
    "What are the sports facilities available?": "Football, basketball, tennis, and indoor sports facilities.",
    "Are there healthcare services on campus?": "Yes, a health clinic is available on campus.",
    "Does the campus have a career center?": "Yes, the career center assists with internships and job placements.",
    "Is there an auditorium for events?": "Yes, the auditorium is used for seminars and events.",
    "Are there accessible facilities for people with disabilities?": "Yes, facilities are accessible for people with disabilities.",
    "Can I host events on campus?": "Yes, event hosting is allowed with prior approval.",
    "Is there a bookstore on campus?": "Yes, the bookstore sells books, merchandise, and supplies.",
    "Are there banking or ATM facilities?": "ATMs are located near the main library and cafeteria.",
    "Is there security on campus?": "Yes, 24/7 security services are available.",
    "What cultural or recreational activities are offered?": "Cultural festivals, concerts, and recreational clubs are offered.",
    "Does the campus have outdoor green spaces?": "Yes, there are outdoor parks and gardens on campus.",
    "Are there on-campus part-time job opportunities?": "Yes, students can work part-time on campus.",
    "Are pets allowed on campus?": "No, pets are not allowed in campus buildings.",
    "What safety measures are in place for students?": "Security cameras, emergency hotlines, and patrols ensure safety.",
    "What sports clubs are available?": "Various sports clubs, including football, basketball, and tennis.",
    "Are there student clubs or societies?": "Yes, there are several academic, cultural, and recreational societies."
    # Add more questions and answers as needed
}


# Preprocess text: tokenization, lowercasing, removing stopwords and punctuation
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

# Preprocess questions in the dictionary
preprocessed_questions = [preprocess_text(q) for q in qa_dict.keys()]

# Route for chatbot interface
@app.route('/', methods=['GET', 'POST'])
def chatbot():
    response = ""
    if request.method == 'POST':
        user_question = request.form['question']
        preprocessed_user_question = preprocess_text(user_question)

        # Use TF-IDF Vectorizer for vectorization
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(preprocessed_questions + [preprocessed_user_question])
        cosine_similarities = cosine_similarity(vectors[-1], vectors[:-1])

        # Find the best match
        best_match_idx = np.argmax(cosine_similarities)
        best_match_score = cosine_similarities[0, best_match_idx]

        # Threshold for similarity
        threshold = 0.3
        if best_match_score >= threshold:
            response = list(qa_dict.values())[best_match_idx]
        else:
            response = "Sorry, I don't understand that question."

    return render_template('chatbot.html', response=response)

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
