// Create three categories of data
let carData = ['type', 'brand', 'fuel', 'drivetrain'];
let datesData = ['first_registration'];
let measurementsData = ['max_speed', 'power', 'weight'];

// Add event listeners to buttons
document.getElementById('car').addEventListener('click', toggleActiveAndUpdate);
document.getElementById('dates').addEventListener('click', toggleActiveAndUpdate);
document.getElementById('measurements').addEventListener('click', toggleActiveAndUpdate);

// Add event listener to reset button
document.getElementById('reset_button').addEventListener('click', function() {
    fetch('/reset_vehicle', {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch((error) => {
        console.error('Error:', error);
    });
	let iframe = document.querySelector('.plate iframe');
	iframe.style.display = 'none';

});

function toggleActiveAndUpdate(event) {
    // Remove .active class from all buttons
    document.getElementById('car').classList.remove('active');
    document.getElementById('dates').classList.remove('active');
    document.getElementById('measurements').classList.remove('active');

    // Add .active class to clicked button
    event.target.classList.add('active');

    // Update displayed data
    updateVehicleData();
}

function updateVehicleData() {
    fetch('/get_vehicle')
        .then(response => response.json())
        .then(vehicle => {
            let activeButton = document.querySelector('.active').id;
            let dataToDisplay = [];

            switch (activeButton) {
                case 'car':
                    dataToDisplay = carData;
                    break;
                case 'dates':
                    dataToDisplay = datesData;
                    break;
                case 'measurements':
                    dataToDisplay = measurementsData;
                    break;
            }

            if (Object.keys(vehicle).length > 0) {
                let html = '';
                for (let key of dataToDisplay) {
                    html += `<p>${key.charAt(0).toUpperCase() + key.slice(1)}: ${vehicle[key]}</p>`;
                }
                document.getElementById('vehicle-data').innerHTML = html;

                // Get the iframe within the div with class .plate
                let iframe = document.querySelector('.plate iframe');

                // If the numberplate number exists, change the src of the iframe
                // If it doesn't exist, hide the iframe
                if (vehicle.license_plate) {
					license = vehicle.license_plate
					console.log(license.replace(/\s/g, ''));
                    iframe.src = `https://git.storbukas.no/personlig-kjennemerke.svg?kjennemerke=${(vehicle.license_plate).replace(/\s/g, '')}`;
                    iframe.style.display = 'flex';
                } else {
                    iframe.style.display = 'none';
                }
            } else {
                document.getElementById('vehicle-data').innerHTML = '<p>No vehicle data available.</p>';
            }
        });
}

// Update vehicle data every 2 seconds
setInterval(updateVehicleData, 2000);