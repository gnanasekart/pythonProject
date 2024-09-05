from github import Github
from config import GITHUB_TOKEN

g = Github(GITHUB_TOKEN)

REPOSITORY_NAME = 'gnanasekart/CodeReview'

def get_pull_request_details(event):
    repo = g.get_repo(REPOSITORY_NAME)
    pr = repo.get_pull(event['pr_number'])
    files_changed = [file.filename for file in pr.get_files()]
    code = "\n".join([file.patch for file in pr.get_files()])
    return {'pr_number': event['pr_number'], 'repository': REPOSITORY_NAME, 'files_changed': files_changed, 'code': code}

def get_related_files(repo_name, pr_number):
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    related_files_content = "\n".join([file.patch for file in pr.get_files()])
    return related_files_content

def post_review_comment(pr_number, comment):
    repo = g.get_repo(REPOSITORY_NAME)
    pr = repo.get_pull(pr_number)
    pr.create_issue_comment(comment)


def get_related_files():
    return None