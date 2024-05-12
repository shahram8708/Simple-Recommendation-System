function getRecommendations() {
    // Get selected role and experience level from the form
    var role = document.getElementById('role').value;
    var experience = document.getElementById('experience').value;

    // Make a POST request to the backend with candidate profile
    fetch('/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            role: role,
            experience_level: experience
        })
    })
    .then(response => response.json())
    .then(data => {
        // Display recommended questions on the frontend
        var recommendationsDiv = document.getElementById('recommendations');
        recommendationsDiv.innerHTML = '<h2>Recommended Questions:</h2>';
        var questionsList = document.createElement('ul');
        data.questions.forEach(question => {
            var listItem = document.createElement('li');
            listItem.textContent = question.question + " - " + question.answer;
            questionsList.appendChild(listItem);
        });
        recommendationsDiv.appendChild(questionsList);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
