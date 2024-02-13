const buttons = document.querySelectorAll('.button_wrapper button');

buttons.forEach((button) => {
	button.addEventListener('click', (event) => {
		// Remove active class from all buttons
		buttons.forEach((btn) => {
			btn.classList.remove('active');
		});

		// Add active class to the clicked button
		event.target.classList.add('active');
	});
});