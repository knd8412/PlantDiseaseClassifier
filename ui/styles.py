styles_css = """
:root {
    --bg-dark: #020617;
    --bg-light: #f3f4f6;
    --text-dark: #020617;
    --text-light: #e5e7eb;
    --card-dark: rgba(15, 23, 42, 0.92);
    --card-light: #ffffff;
    --border-dark: rgba(148, 163, 184, 0.5);
    --border-light: rgba(209, 213, 219, 1);
    --muted-dark: #9ca3af;
    --muted-light: #6b7280;
}

@media (prefers-color-scheme: dark) {
    body {
        background-color: var(--bg-dark);
        color: var(--text-light);
    }
}

@media (prefers-color-scheme: light) {
    body {
        background-color: var(--bg-light);
        color: var(--text-dark);
    }
}

.gradio-container {
    min-height: 100vh;
    background:
        radial-gradient(circle at 0% 0%, #22c55e33 0, transparent 45%),
        radial-gradient(circle at 100% 100%, #0ea5e933 0, transparent 55%),
        radial-gradient(circle at 50% 10%, #910ee955 0, transparent 45%),
        var(--bg-dark);
    font-family: system-ui, sans-serif;
    margin: 0;
}

#app-title, #app-subtitle {
    text-align: center;
    color: var(--text-light);
}

#app-title {
    font-size: 2rem;
    font-weight: 300;
    margin-bottom: 0.4rem;
}

#app-subtitle {
    font-size: 1.5rem;
    margin-bottom: 1.75rem;
}

.card {
    background: var(--card-dark);
    border-radius: 16px;
    border: 1px solid var(--border-dark);
    padding: 1.25rem;
    box-shadow: 0 16px 35px rgba(15, 23, 42, 0.8);
}

.card .gr-image {
    border-radius: 12px;
    border: 1px solid rgba(200, 163, 184, 0.4);
    overflow: hidden;
}

#analyse_btn {
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    font-weight: 600;
    padding: 0.7rem 1.6rem;
    border: none;
    border-radius: 999px;
    box-shadow: 0 14px 30px rgba(88, 80, 236, 0.2);
    transition: transform 0.1s ease, box-shadow 0.1s ease;
}

#analyse_btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 38px rgba(88, 80, 236, 0.3);
    filter: brightness(1.05);
}

#footer-note {
    font-size: 0.95rem;
    color: var(--text-light);
    margin-top: 1rem;
}

.gradio-container table {
    font-size: 0.9rem;
    border-collapse: collapse;
    width: 100%;
}

.gradio-container table thead th {
    background-color: rgba(15, 23, 42, 0.9);
    padding: 0.5rem;
    border-bottom: 1px solid var(--border-dark);
}

.gradio-container table tbody td {
    background-color: transparent;
    padding: 0.5rem;
    border-bottom: 1px solid rgba(30, 41, 59, 0.5);
}

"""
