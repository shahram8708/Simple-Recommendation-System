from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app = Flask(__name__, static_folder='static')

# Define a class to represent interview questions
class InterviewQuestion:
    def __init__(self, question, tags, difficulty, answer):
        self.question = question
        self.tags = tags
        self.difficulty = difficulty
        self.answer = answer

# Define a function to recommend questions based on candidate attributes
def recommend_questions(candidate_profile, questions):
    recommended_questions = []
    for question in questions:
        if candidate_profile["role"] in question.tags:
            if candidate_profile["experience_level"] == "Entry" and question.difficulty in ["Easy", "Medium"]:
                recommended_questions.append(question)
            elif candidate_profile["experience_level"] == "Intermediate" and question.difficulty in ["Medium", "Hard"]:
                recommended_questions.append(question)
            elif candidate_profile["experience_level"] == "Senior" and question.difficulty in ["Hard"]:
                recommended_questions.append(question)
    return recommended_questions

# Define sample candidate attributes
candidate_profile = {"role": "Software Engineer", "experience_level": "Intermediate"}

# Define sample interview questions with answers
sample_questions = [
    InterviewQuestion("What is OOP?", ["Software Engineer", "Data Scientist"], "Easy", "Object-oriented programming (OOP) is a programming paradigm based on the concept of 'objects', which can contain data, in the form of fields (often known as attributes), and code, in the form of procedures (often known as methods)."),
    InterviewQuestion("What is SQL?", ["Software Engineer", "Data Scientist"], "Easy", "Structured Query Language (SQL) is a domain-specific language used in programming and designed for managing data held in a relational database management system (RDBMS) or for stream processing in a relational data stream management system (RDSMS)."),
    InterviewQuestion("Explain Gradient Descent", ["Data Scientist"], "Medium", "Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient."),
    InterviewQuestion("Describe Agile methodology", ["Product Manager"], "Medium", "Agile methodology is a set of principles for software development under which requirements and solutions evolve through the collaborative effort of self-organizing cross-functional teams."),
    InterviewQuestion("What is Big O notation?", ["Software Engineer"], "Medium", "Big O notation is a mathematical notation used in computer science to describe the performance or complexity of an algorithm."),
    InterviewQuestion("What is A/B testing?", ["Product Manager"], "Hard", "A/B testing is a randomized experiment with two variants, A and B, which are the control and treatment in the controlled experiment. It is a form of statistical hypothesis testing."),
    InterviewQuestion("What is linear regression?", ["Data Scientist"], "Medium", "Linear regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables."),
    InterviewQuestion("What is backpropagation?", ["Data Scientist"], "Hard", "Backpropagation is a supervised learning algorithm used for training artificial neural networks."),
    InterviewQuestion("What is the difference between supervised and unsupervised learning?", ["Data Scientist"], "Medium", "Supervised learning is a type of machine learning where the model is trained on a labeled dataset, while unsupervised learning is trained on an unlabeled dataset."),
    InterviewQuestion("Explain decision trees.", ["Data Scientist"], "Medium", "Decision trees are a supervised learning method used for classification and regression tasks. They work by recursively splitting the data into subsets based on the value of attributes."),
    InterviewQuestion("What is the bias-variance tradeoff?", ["Data Scientist"], "Hard", "The bias-variance tradeoff is a fundamental concept in machine learning that refers to the balance between a model's ability to capture the underlying patterns in the data (bias) and its flexibility to adapt to new, unseen data (variance)."),
    InterviewQuestion("What is overfitting?", ["Data Scientist"], "Medium", "Overfitting occurs when a machine learning model learns the training data too well, capturing noise or random fluctuations in the data instead of the underlying patterns."),
    InterviewQuestion("What is regularization?", ["Data Scientist"], "Hard", "Regularization is a technique used to prevent overfitting in machine learning models by adding a penalty term to the loss function that penalizes large parameter values."),
    InterviewQuestion("What is natural language processing (NLP)?", ["Data Scientist"], "Medium", "Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language."),
    InterviewQuestion("What is deep learning?", ["Data Scientist"], "Hard", "Deep learning is a subset of machine learning that uses neural networks with many layers (deep neural networks) to learn from large amounts of data."),
    InterviewQuestion("What is a convolutional neural network (CNN)?", ["Data Scientist"], "Hard", "A convolutional neural network (CNN) is a type of deep neural network used for image recognition and classification. It consists of multiple layers of convolutional and pooling operations."),
    InterviewQuestion("What is a recurrent neural network (RNN)?", ["Data Scientist"], "Hard", "A recurrent neural network (RNN) is a type of neural network designed for sequence modeling tasks, such as natural language processing and time series prediction. It has loops that allow information to persist over time."),
    InterviewQuestion("What is reinforcement learning?", ["Data Scientist"], "Hard", "Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment to achieve some goal."),
    InterviewQuestion("What is unsupervised learning?", ["Data Scientist"], "Medium", "Unsupervised learning is a type of machine learning where the model is trained on an unlabeled dataset and must learn the underlying structure of the data."),
    InterviewQuestion("What is k-means clustering?", ["Data Scientist"], "Medium", "K-means clustering is a popular unsupervised learning algorithm used for clustering tasks. It partitions the data into k clusters based on the similarity of data points."),
    InterviewQuestion("What is dimensionality reduction?", ["Data Scientist"], "Hard", "Dimensionality reduction is the process of reducing the number of features in a dataset while preserving its important structure and relationships."),
    InterviewQuestion("What is principal component analysis (PCA)?", ["Data Scientist"], "Medium", "Principal component analysis (PCA) is a technique used for dimensionality reduction. It transforms the data into a new coordinate system such that the greatest variance lies on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on."),
    InterviewQuestion("What is ensemble learning?", ["Data Scientist"], "Medium", "Ensemble learning is a machine learning technique where multiple models are trained to solve the same problem and their predictions are combined to improve the overall performance."),
    InterviewQuestion("What is deep reinforcement learning?", ["Data Scientist"], "Hard", "Deep reinforcement learning is a combination of deep learning and reinforcement learning techniques, where deep neural networks are used to approximate the value functions or policies in reinforcement learning tasks."),
    InterviewQuestion("What is transfer learning?", ["Data Scientist"], "Medium", "Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a second task."),
    InterviewQuestion("What is generative adversarial networks (GANs)?", ["Data Scientist"], "Hard", "Generative adversarial networks (GANs) are a type of deep learning model used for generating new data samples that are similar to a given dataset."),
    InterviewQuestion("What is support vector machines (SVM)?", ["Data Scientist"], "Hard", "Support vector machines (SVM) are supervised learning models used for classification and regression analysis."),
    InterviewQuestion("What is the difference between bagging and boosting?", ["Data Scientist"], "Medium", "Bagging and boosting are ensemble learning techniques used to improve the performance of machine learning models. Bagging involves training multiple models independently and combining their predictions, while boosting involves sequentially training models to correct the errors of the previous models."),
    InterviewQuestion("What is the curse of dimensionality?", ["Data Scientist"], "Hard", "The curse of dimensionality refers to the phenomenon where the performance of machine learning algorithms deteriorates as the number of features or dimensions increases. It becomes increasingly difficult to find patterns and make accurate predictions in high-dimensional spaces."),
    InterviewQuestion("What is time series analysis?", ["Data Scientist"], "Medium", "Time series analysis is a statistical technique used to analyze time-ordered data points. It is commonly used in fields such as finance, economics, and signal processing to forecast future values based on past observations."),
    InterviewQuestion("What is the difference between classification and regression?", ["Data Scientist"], "Medium", "Classification and regression are two types of supervised learning tasks. Classification involves predicting a categorical label or class, while regression involves predicting a continuous value."),
    InterviewQuestion("What is the bias-variance tradeoff?", ["Data Scientist"], "Hard", "The bias-variance tradeoff is a fundamental concept in machine learning that refers to the balance between a model's ability to capture the underlying patterns in the data (bias) and its flexibility to adapt to new, unseen data (variance)."),
    InterviewQuestion("What is ensemble learning?", ["Data Scientist"], "Medium", "Ensemble learning is a machine learning technique where multiple models are trained to solve the same problem and their predictions are combined to improve the overall performance."),
    InterviewQuestion("What is deep reinforcement learning?", ["Data Scientist"], "Hard", "Deep reinforcement learning is a combination of deep learning and reinforcement learning techniques, where deep neural networks are used to approximate the value functions or policies in reinforcement learning tasks."),
    InterviewQuestion("What is transfer learning?", ["Data Scientist"], "Medium", "Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a second task."),
    InterviewQuestion("What is generative adversarial networks (GANs)?", ["Data Scientist"], "Hard", "Generative adversarial networks (GANs) are a type of deep learning model used for generating new data samples that are similar to a given dataset."),
    InterviewQuestion("What is support vector machines (SVM)?", ["Data Scientist"], "Hard", "Support vector machines (SVM) are supervised learning models used for classification and regression analysis."),
    InterviewQuestion("What is the difference between bagging and boosting?", ["Data Scientist"], "Medium", "Bagging and boosting are ensemble learning techniques used to improve the performance of machine learning models. Bagging involves training multiple models independently and combining their predictions, while boosting involves sequentially training models to correct the errors of the previous models."),
    InterviewQuestion("What is the curse of dimensionality?", ["Data Scientist"], "Hard", "The curse of dimensionality refers to the phenomenon where the performance of machine learning algorithms deteriorates as the number of features or dimensions increases. It becomes increasingly difficult to find patterns and make accurate predictions in high-dimensional spaces."),
    InterviewQuestion("What is time series analysis?", ["Data Scientist"], "Medium", "Time series analysis is a statistical technique used to analyze time-ordered data points. It is commonly used in fields such as finance, economics, and signal processing to forecast future values based on past observations."),
    InterviewQuestion("What is the difference between classification and regression?", ["Data Scientist"], "Medium", "Classification and regression are two types of supervised learning tasks. Classification involves predicting a categorical label or class, while regression involves predicting a continuous value."),
    InterviewQuestion("What is the bias-variance tradeoff?", ["Data Scientist"], "Hard", "The bias-variance tradeoff is a fundamental concept in machine learning that refers to the balance between a model's ability to capture the underlying patterns in the data (bias) and its flexibility to adapt to new, unseen data (variance)."),
    InterviewQuestion("What is ensemble learning?", ["Data Scientist"], "Medium", "Ensemble learning is a machine learning technique where multiple models are trained to solve the same problem and their predictions are combined to improve the overall performance."),
    InterviewQuestion("What is deep reinforcement learning?", ["Data Scientist"], "Hard", "Deep reinforcement learning is a combination of deep learning and reinforcement learning techniques, where deep neural networks are used to approximate the value functions or policies in reinforcement learning tasks."),
    InterviewQuestion("What is transfer learning?", ["Data Scientist"], "Medium", "Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a second task."),
    InterviewQuestion("What is generative adversarial networks (GANs)?", ["Data Scientist"], "Hard", "Generative adversarial networks (GANs) are a type of deep learning model used for generating new data samples that are similar to a given dataset."),
    InterviewQuestion("What is support vector machines (SVM)?", ["Data Scientist"], "Hard", "Support vector machines (SVM) are supervised learning models used for classification and regression analysis."),
    InterviewQuestion("What is the difference between bagging and boosting?", ["Data Scientist"], "Medium", "Bagging and boosting are ensemble learning techniques used to improve the performance of machine learning models. Bagging involves training multiple models independently and combining their predictions, while boosting involves sequentially training models to correct the errors of the previous models."),
    InterviewQuestion("What is the curse of dimensionality?", ["Data Scientist"], "Hard", "The curse of dimensionality refers to the phenomenon where the performance of machine learning algorithms deteriorates as the number of features or dimensions increases. It becomes increasingly difficult to find patterns and make accurate predictions in high-dimensional spaces."),
    InterviewQuestion("What is time series analysis?", ["Data Scientist"], "Medium", "Time series analysis is a statistical technique used to analyze time-ordered data points. It is commonly used in fields such as finance, economics, and signal processing to forecast future values based on past observations."),
    InterviewQuestion("What is the difference between classification and regression?", ["Data Scientist"], "Medium", "Classification and regression are two types of supervised learning tasks. Classification involves predicting a categorical label or class, while regression involves predicting a continuous value."),
]

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # Receive candidate profile from frontend
    candidate_profile = request.json
    # Call recommend_questions function with sample questions
    recommended_questions = recommend_questions(candidate_profile, sample_questions)
    # Create a list of dictionaries containing question and answer
    response = [{"question": question.question, "answer": question.answer} for question in recommended_questions]
    # Return recommended questions and answers as JSON
    return jsonify({"questions": response})

if __name__ == '__main__':
    app.run(debug=True)
