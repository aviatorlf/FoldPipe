from google_auth_oauthlib.flow import InstalledAppFlow

print("Opening browser for Google Authentication...")
flow = InstalledAppFlow.from_client_secrets_file(
    'gcp_secret_dataset/client_secret_496839852310-tin5b2grsp8uccbou0bf9m5shct743tj.apps.googleusercontent.com.json', 
    scopes=['https://www.googleapis.com/auth/drive']
)
creds = flow.run_local_server(port=0)

with open('gcp_secret_dataset/token.json', 'w') as token:
    token.write(creds.to_json())
    
print("Headless token generated successfully in gcp_secret_dataset/token.json!")
