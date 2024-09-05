from github_api import get_related_files

def fetch_contextual_data(pr_details):
    related_files_content = get_related_files(pr_details['repository'], pr_details['pr_number'])
    context = f"Here are related files:\n{related_files_content}"
    return context