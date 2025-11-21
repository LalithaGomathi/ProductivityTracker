
# ServiceNow connector stub
# Requires: requests
# Configure with your instance, user, and token/password.
import requests

class ServiceNowConnector:
    def __init__(self, instance_url, user, pwd):
        self.base = instance_url.rstrip('/')
        self.auth = (user, pwd)
    def query_incidents(self, since=None, limit=100):
        url = f"{self.base}/api/now/table/incident"
        params = {'sysparm_limit': limit}
        if since: params['sysparm_query'] = f"sys_created_on>={since}"
        resp = requests.get(url, auth=self.auth, params=params)
        resp.raise_for_status()
        data = resp.json()
        # implement mapping to CSV rows as needed
        return data
