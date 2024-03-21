const fileInput = document.getElementById('file-input');
const fileLabel = document.querySelector('.file-label');

fileInput.addEventListener('change', () => {
  const files = fileInput.files;
  if (files.length > 0) {
    fileLabel.innerHTML = `${files.length} file(s) selected`;
  } else {
    fileLabel.innerHTML = 'Choose files';
  }
});
const contactLink = document.querySelector('.contact-link');
const contactInfo = document.querySelector('.contact-info');

contactLink.addEventListener('click', (event) => {
  event.preventDefault();
  contactInfo.style.display = 'block';
});