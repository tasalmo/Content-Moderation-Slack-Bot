import json
import boto3
import urllib
import os
import logging
import csv

rekognition = boto3.client('rekognition') #set up connection to rekognition
comprehend = boto3.client('comprehend')#set up connection to comprehend
s3 = boto3.client('s3')#set up connection to s3

model_arn_toxic = 'arn:aws:comprehend:us-east-2:114577848449:document-classifier-endpoint/ToxicityIdentifier2' # Specify the ARN of your custom classification model

SUPPORTED_TYPES = ['jpeg', 'jpg', 'png', 'mov', 'mp4']  #specify file types rekognition can support

def get_secret(): #retreiving stored variables from secrets manager
    secret_name = "DetectModerationLabels-secrets"
    region_name = "us-east-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    return json.loads(secret)

secrets = get_secret() #retrieve secrets from secrets manager
VERIFICATION_TOKEN = secrets.get("VERIFICATION_TOKEN")  # Slack verification token from environment variables
ACCESS_TOKEN_USER = secrets.get("ACCESS_TOKEN_USER")  # Slack OAuth user access token from environment variables
ACCESS_TOKEN_BOT = secrets.get("ACCESS_TOKEN_BOT") # Slack OAuth bot access token from environment variables
MIN_CONFIDENCE = float(secrets.get("CONFIDENCE"))  # minimum confidence level from environment variables

response_payload = { #send a response back to slack that request is received
    "statusCode": 200,
    "body": "OK"
}

def lambda_handler(event, context):
    #extract the slack event from the client request json
    slack_body = event.get("body")
    slack_event = json.loads(slack_body).get("event")
    
    if json.loads(slack_body).get('challenge') is not None:  # Respond to Slack event subscription URL verification challenge
        print('Presented with URL verification challenge- responding accordingly...')
        challenge = json.loads(slack_body).get("challenge")
        return {
            'statusCode': 200,
            'body': challenge
        }
    send_ok()
    
    # to see what the slack event is in cloudwatch
    print("event is: " + str(event))

    #check if bot requests and ignore if so
    if slack_event.get("bot_id") is not None:
        return response_payload
    if slack_event.get("subtype") == "message_deleted":
        return response_payload

    #get time stamp of slack event
    ts = slack_event.get("event_ts")
    
    #log cloudwatch errror
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    
    # Create a CloudWatchLogs client
    cloudwatch_logs = boto3.client('logs')
    
    print(slack_event)
    
    try:
        if not verify_token(slack_body): #verify that the token sent through the post request is same as the slack app token
            return response_payload
        
        #channel id and time stamp of message/image sent in the chat
        channel_id = slack_event.get("channel")
        
        #  1. if text: check if text contains violations
        if "text" in slack_event and slack_event["text"] != '':
            # Perform actions for new message
            text = slack_event.get("text") #get text from slack chat
            
            #first check if text contains any profanity using the dataset
            profanity_result = moderate_text_profanity(text, channel_id, ts)
            if profanity_result != False:
                delete_and_warn(profanity_result, channel_id, ts)
                return response_payload
            #then check with comprehend if text is toxic
            toxicity_result = moderate_text_toxicity(text, channel_id, ts) #send text to comprehend
            if toxicity_result != False:
                delete_and_warn(toxicity_result, channel_id, ts)
                return response_payload
            
            #check if it is both text and image, if just text then exit
            if "files" not in slack_event or not slack_event["files"]:
                return response_payload

        #  2. if media, moderate media
        moderate_media(slack_event, channel_id) #if not text, then start image moderation process
        
        raise Exception("Something went wrong!")
    
    except Exception as e:
        # Log the error message to CloudWatch
        logger.error(f"Error: {str(e)}")
    
    return response_payload

def send_ok():
    return response_payload
    
def verify_token(slack_body): #verify that the token from the event message matches the slack app verification token
    if json.loads(slack_body).get("token") != VERIFICATION_TOKEN:
        return False
    return True
  
def moderate_text_profanity(text, channel_id, ts):
    #Read the CSV file and store the words in a list
    csv_file_path = "profanity.csv"
    
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        words_list = [row[0] for row in reader]
    
    #Take text input from the user and split it into individual words
    input_words = text.lower().split()
    
    #use 'any()' to check if any word matches
    is_match = any(word in input_words for word in words_list)
    
    #output the result
    if is_match:
        message = "Profanity"
        return message
    else:
        return False
 
def moderate_text_toxicity(text, channel_id, ts):
    response = comprehend.classify_document(
        Text = text,
        EndpointArn = model_arn_toxic
    )
                
    #get the class (toxic or not toxic)
    classes = response['Classes']
        
    # Return the top class and its confidence score
    top_class = classes[0]['Name']
    confidence = round((classes[0]['Score'] * 100), 2)
                
    if top_class == "Toxic":
        message = str(top_class) + ", Confidence Score: " + str(confidence) + "%"
        return message
    else: 
        return False

def moderate_media(slack_event, channel_id):
    file_details = slack_event['files'][0] #get image file
        
    if not validate_media(slack_event, file_details): #check if a supported image/video file is sent
        return response_payload
        
    file_id = file_details['id'] # get id of file

    image_bytes = get_image_bytes(file_details) # get bytes of an image
        
    label_count, label_name = rekognition_moderation(image_bytes) #call rekognition api and check how many explicit labels come back on the image 
            
    if label_count > 0: #if image is explicit then delete it and post a warning message in chat
        delete_image(file_id)
        message = "*Warning*: Previous image sent was deleted due to violating moderation standards. Violations include: " + str(label_name)
        post_message(channel_id, message) #post message detailing that the image was deleted because of its explicit labels

def delete_and_warn(result, channel_id, ts):
    delete_text(channel_id, ts)
    message = "*Warning*: Previous message sent was deleted due to violating moderation standards. Violations include:\n • " + str(result)
    post_message(channel_id, message) #post message detailing that the image was deleted because of its explicit labels
    
def validate_media(slack_event, file_details): 
    file_subtype = slack_event.get("subtype")

    if file_subtype != 'file_share':
        return False
        
    mime_type = file_details['filetype']
    
    if mime_type in [ 'mp4', 'mov']: #if video
        moderate_video(slack_event)
        return response_payload
        
    if mime_type not in SUPPORTED_TYPES:
        file_id = file_details['id']
        channel_id = slack_event.get("channel")
        message = "*Warning*: Previous file sent was deleted as it is an unsupported file type. Supported file types are: JPEG, JPG, PNG, MP4, MOV"
        # Delete the image
        delete_image(file_id)
        # Send the warning message to the Slack channel
        post_message(channel_id, message)
    return True

def moderate_video(slack_event):
    #name of your S3 bucket where the video is stored
    s3_bucket = 'video-content-moderation'
    #key (filename) of the video object in your S3 bucket
    video_key = slack_event['files'][0]['title']
    #get url from slack event
    video_url = slack_event['files'][0]['url_private']
    #filename of video in the /tmp directory in lambda
    local_filename = '/tmp/video.mp4'

    try:
        # Download the video from the URL
        headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN_BOT}'
        }
    
        try:
            req = urllib.request.Request(video_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    with open(local_filename, 'wb') as f:
                        f.write(response.read())
                    print("Video downloaded successfully.")
                else:
                    print(f"Failed to download video. Status code: {response.status}")
        except urllib.error.URLError as e:
            print(f"An error occurred: {e}")
            return response_payload
        s3.upload_file(local_filename, s3_bucket, video_key)
    except Exception as e:
        return response_payload
    # Start the CM on the video
    response = rekognition.start_content_moderation(
        Video={
            'S3Object': {
                'Bucket': s3_bucket,
                'Name': video_key
            }
        },
        NotificationChannel={
            'SNSTopicArn': 'arn:aws:sns:us-east-2:114577848449:AmazonRekognition',  #SNS topic to receive completion notifications
            'RoleArn': 'arn:aws:iam::114577848449:role/Admin'        #IAM role that allows Rekognition to publish to SNS
        }
    )

    # Get the JobId from the response
    job_id = response['JobId']
    
    # Poll the job status and wait for the analysis to complete
    while True:
        job_response = rekognition.get_content_moderation(JobId=job_id)
        if job_response['JobStatus'] == 'SUCCEEDED':
            break
        elif job_response['JobStatus'] == 'FAILED':
            raise Exception('Face detection job failed.')

    # Retrieve the results
    moderation_results = job_response
    
    os.remove(local_filename) # delete the saved video in lambda
    
    label_name = ""
    label_list = []
    for label in moderation_results['ModerationLabels']: #make a list of violations detected
        if label['ModerationLabel']['Name'] not in label_list:
            label_list.append(label['ModerationLabel']['Name'])
            label_name += "\n• " + (label['ModerationLabel']['Name']) + ", Confidence Score: " + str(round((label['ModerationLabel']['Confidence']), 2)) + "% "

    file_details = slack_event['files'][0] #get video file
    file_id = file_details['id'] # get id of file
    channel_id = slack_event.get("channel")
    
    if len(moderation_results['ModerationLabels']) > 0: #if there are any moderation labels detected
        print(label_name)
        delete_image(file_id)
        message = "*Warning*: Previous video sent was deleted due to violating moderation standards. Violations include: " + str(label_name)
        post_message(channel_id, message) #post message detailing that the image was deleted because of its explicit labels
       
def get_image_bytes(file_details): #returns the bytes of an image
    url = file_details['url_private']
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer %s' % ACCESS_TOKEN_BOT})
    return urllib.request.urlopen(request).read()
    
def rekognition_moderation(image_bytes): #calls rekognition api and returns the number of explicit labels returned
    
    response = rekognition.detect_moderation_labels(
        Image={
            'Bytes': image_bytes,
        }, 
        MinConfidence = MIN_CONFIDENCE
    )

    label_name = ""
    for label in response['ModerationLabels']:
        label_name += "\n• " + (label['Name']) + ", Confidence Score: " + str(round((label['Confidence']), 2)) + "% "

    return len(response['ModerationLabels']), label_name

def delete_image(file_id):
    data = urllib.parse.urlencode(
        (
            ("token", ACCESS_TOKEN_USER),
            ("file", file_id)
        )
    )
    data = data.encode("ascii")
    SLACK_URL = 'https://slack.com/api/files.delete'
    request = urllib.request.Request(SLACK_URL, data=data, method="POST")
    request.add_header( "Content-Type", "application/x-www-form-urlencoded" )
                    
    # Fire off the request!
    urllib.request.urlopen(request)

def delete_text(channel, ts):
    data = urllib.parse.urlencode(
        (
            ("token", ACCESS_TOKEN_USER),
            ("channel", channel),
            ("ts", ts)
        )
    )
    data = data.encode("ascii")
    SLACK_URL = 'https://slack.com/api/chat.delete'
    request = urllib.request.Request(SLACK_URL, data=data, method="POST")
    request.add_header( "Content-Type", "application/x-www-form-urlencoded" )
                    
    # Fire off the request!
    urllib.request.urlopen(request)
    
def post_message(channel_id, message): #posts a message to the channel that the explicit image was sent, and details what explicit labels the image contains 
    data = urllib.parse.urlencode(
        (
            ("token", ACCESS_TOKEN_BOT),
            ("channel", channel_id),
            ("text", message)
        )
    )
    data = data.encode("ascii")
    SLACK_URL = "https://slack.com/api/chat.postMessage"
    request = urllib.request.Request(SLACK_URL, data=data, method="POST")
    request.add_header( "Content-Type", "application/x-www-form-urlencoded" )
                    
    # Fire off the request!
    urllib.request.urlopen(request)