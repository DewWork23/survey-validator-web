// Main JavaScript file for Survey Validator

document.addEventListener('DOMContentLoaded', function() {
    // Set default dates for validation form
    const today = new Date();
    const threeMonthsAgo = new Date();
    threeMonthsAgo.setMonth(today.getMonth() - 3);

    const startDateInput = document.getElementById('start_date');
    const endDateInput = document.getElementById('end_date');

    if (startDateInput && endDateInput) {
        startDateInput.value = threeMonthsAgo.toISOString().split('T')[0];
        endDateInput.value = today.toISOString().split('T')[0];
    }

    // Form validation
    const validationForm = document.getElementById('validation-form');
    if (validationForm) {
        validationForm.addEventListener('submit', function(event) {
            const communityFile = document.getElementById('community_file');
            const incentiveFile = document.getElementById('incentive_file');
            const startDate = document.getElementById('start_date');
            const endDate = document.getElementById('end_date');

            let isValid = true;
            let errorMessage = '';

            // Check if files are selected
            if (!communityFile.files.length) {
                isValid = false;
                errorMessage += 'Please select a community survey file.\n';
            }
            if (!incentiveFile.files.length) {
                isValid = false;
                errorMessage += 'Please select an incentive survey file.\n';
            }

            // Check if dates are valid
            if (startDate.value && endDate.value) {
                const start = new Date(startDate.value);
                const end = new Date(endDate.value);
                if (start > end) {
                    isValid = false;
                    errorMessage += 'Start date must be before end date.\n';
                }
            }

            if (!isValid) {
                event.preventDefault();
                alert(errorMessage);
            }
        });
    }

    // File input change handlers
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const fileName = this.files[0]?.name;
            const fileLabel = this.nextElementSibling;
            if (fileLabel && fileName) {
                fileLabel.textContent = fileName;
            }
        });
    });

    // Add loading state to submit button
    const submitButton = document.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.addEventListener('click', function() {
            if (validationForm.checkValidity()) {
                this.disabled = true;
                this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            }
        });
    }
}); 