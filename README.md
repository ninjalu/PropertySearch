# PropertySearch (In progress)
The aim of this poject is to create a recommender system for new listings on Rightmove based on image embeddings of exterior, interior and garden preference (content).

## 1. Data collection:
* Images were collected for all properties listed for London on a single day.
* Property information on location, price, and description was collected.

## 2. Classification:
* Classify all images into three categories: exterior, interior and garden using ResNet transfer learning (acc 95% on validation)

## 3. Embeddings:
* Using autoencoder to find embeddings of images in all three categories
* Create an image bank where users can click through a few images to generate user embeddings based on average of image embeddings.

## 4. Web app:
* Front end web app where users interact (list, dislike) images.
* Sign up for updates whenever similar property comes on to the market in the given area.

## Further development:
* Monitor user interaction and possibly build a colaborative filtering system.

