document.addEventListener('DOMContentLoaded', function () {
    const root = document.documentElement;
    const themeToggleBtn = document.getElementById('themeToggleBtn');

    const savedTheme = localStorage.getItem('theme') || 'dark';
    root.classList.add(`${savedTheme}-theme`);

    if (themeToggleBtn) {
        updateButtonIcon(savedTheme);

        themeToggleBtn.addEventListener('click', function () {
            const currentTheme = root.classList.contains('dark-theme') ? 'dark' : 'light';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            root.classList.remove(`${currentTheme}-theme`);
            root.classList.add(`${newTheme}-theme`);
            localStorage.setItem('theme', newTheme);
            updateButtonIcon(newTheme);
        });
    }

    function updateButtonIcon(theme) {
        if (themeToggleBtn) {
            themeToggleBtn.textContent = theme === 'dark' ? '🌙' : '☀️';
        }
    }
});
