import requests

sources = open("TomatoURLs.txt",'r')
print(sources.read())

img_data = requests.get("http://farm3.static.flickr.com/2370/2054573289_62c4e4b029.jpg").content
with open('image_name.jpg', 'wb') as handler:
    handler.write(img_data)

