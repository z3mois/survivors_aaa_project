from locust import HttpUser, TaskSet, task, between

class MyUserBehavior(TaskSet):
    @task
    def search_candidates(self):
        """
            create task for user
        """
        payload = {
            "description": "Требования к вакансии",
            "city": "Чита",
            "top_k": 10
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        
        response = self.client.get("http://localhost:8501", data=payload, headers=headers)
        print(response.status_code)
        print(response.text)
class MyUser(HttpUser):
    """
        create user
    """
    tasks = [MyUserBehavior]
    wait_time = between(1, 5)  # Время ожидания между запросами
