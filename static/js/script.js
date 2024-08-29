document.addEventListener("DOMContentLoaded", function() {
    const articles = document.querySelectorAll('.article-link');

    articles.forEach(function(article) {
        const url = article.getAttribute('href');
        const articleTitle = extractTitleFromURL(url);
        article.querySelector('.article-title').textContent = articleTitle;
        article.setAttribute('data-title', articleTitle);
    });

    function extractTitleFromURL(url) {
        // Remove the file extension (.html)
        const cleanUrl = url.replace('.html', '');
        // Replace hyphens or underscores with spaces and capitalize words
        const title = cleanUrl.split('/').pop().replace(/[-_]/g, ' ')
            .replace(/\b\w/g, letter => letter.toUpperCase());
        return title;
    }
});
