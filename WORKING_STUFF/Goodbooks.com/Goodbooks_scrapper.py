import os
import re
import pandas as pd
import scrapy
import requests
from scrapy.crawler import CrawlerProcess

class GoodReadsSpider(scrapy.Spider):
	name = 'goodreads_spider'
	download_delay = 0.25 #Used to make the downloader wait before downloading data from another page!!

	def start_requests(self):
		for page_id in range(START_PAGE, END_PAGE+1):
			page_url = f'https://www.goodreads.com/list/show/1.Best_Books_Ever?page={page_id}'  #input of the book using the URL!!
			yield scrapy.Request(url=page_url, callback=self.parse_page)
	

	def parse_page(self, selector):
		all_book_titles = selector.css('a.bookTitle')
		for curr_book in all_book_titles:
			book_link = curr_book.attrib['href']
			book_url = f'https://www.goodreads.com{book_link}'
			yield selector.follow(url=book_url, callback=self.parse_book)
	

	#Defining on what should be extracted!!
	def parse_book(self, selector):

		goodbooks_book_id = None
		book_title = None
		book_orig_title = None 
		book_series = None
		book_language = None
		book_authors = []
		book_avg_rating = None
		book_num_ratings = None
		book_num_reviews = None
		book_genres = []
		book_description = ''
		image_url = ''


		image_url = selector.css('#coverImage ::attr(src)').get()
		print(image_url)

		#EXTRACTION OF DATA INDIVIDUALLY!!
		book_url = selector.css('#imagecol > div.bookCoverContainer > div.bookCoverPrimary > a::attr(href)').get()
		digits = [d for d in book_url if d.isdigit()]
		goodbooks_book_id = ''.join(digits)
		

		
		book_title = selector.css('h1#bookTitle::text').get()
		book_title = None if book_title is None else book_title.strip()
		

		
	
		
		book_data = selector.css('#bookDataBox > div.clearFloats')
		for data in book_data:
			
			row_title = data.css('.infoBoxRowTitle::text').get()

			
			if 'Original Title' == row_title:
				book_orig_title = data.css('.infoBoxRowItem::text').get()
				book_orig_title = None if book_orig_title is None else book_orig_title.strip() 
			

			if 'Series' == row_title:
				book_series = data.css('.infoBoxRowItem > a::text').get()
				hash_idx = book_series.find('#')
				if hash_idx != -1:
					book_series = book_series[:hash_idx-1]
			

			
			if 'Edition Language' == row_title:
				book_language = data.css('.infoBoxRowItem::text').get()
				book_language = None if book_language is None else book_language.strip()
	
	
		for author in selector.css('.authorName__container'):
			author_name = author.css('a > span::text').get()
			author_role = author.css('.role::text').get()
			
			book_authors.append(author_name if author_role is None else ' '.join([author_name, author_role]))
	
		
		book_avg_rating = selector.css('span[itemprop="ratingValue"]::text').get()
		if book_avg_rating != None:
			book_avg_rating = book_avg_rating.strip()

		
		book_metas = selector.css('#bookMeta > a::text').getall()
		
		num_ratings_idx = [i for i,s in enumerate(book_metas) if ('ratings' in s or 'rating' in s)]
		if len(num_ratings_idx):
			book_num_ratings = book_metas[num_ratings_idx[0]].replace('\n', '').replace(',', '').replace('ratings', '').replace('rating', '').strip()
		
		num_reviews_idx = [i for i,s in enumerate(book_metas) if ('reviews' in s or 'review' in s)]
		if len(num_reviews_idx):
			book_num_reviews = book_metas[num_reviews_idx[0]].replace('\n', '').replace(',', '').replace('reviews', '').replace('review', '').strip()
		
	
		book_genres = selector.css('div.left > a.bookPageGenreLink::text').getall()
		
		
		desc_texts = self.process_book_description(selector.xpath('//*[@id="description"]/span[contains(@style, "display:none")]/node()').getall())
		book_description = ' '.join(desc_texts)
		


		#IF THE DATA IS NOT PRESENT WE NEED TO NULLIFY THAT PARTICUALAR DATA!! and throw a error message!!

		if not len(book_description):
			desc_texts = self.process_book_description(selector.xpath('//*[@id="description"]/span/node()').getall())
			book_description = ' '.join(desc_texts)
		
		if None == goodbooks_book_id:
			print(f'Error: Missing goodbooks ID ({selector.url}).')
			return
		if None == book_title:
			print(f'Error: Missing book title ({selector.url}).')
			return
		
		if None == book_orig_title:
			print(f'Warning: Missing book original title ({selector.url}).')
		
		if None == book_series:
			print(f'Warning: Book missing series ({selector.url}).')
		
		if None == book_language:
			print(f'Warning: Book missing language ({selector.url}).')
		
		if not len(book_authors):
			print(f'Warning: Book missing authors ({selector.url}).')
		
		if None == book_avg_rating:
			print(f'Warning: Book missing average rating ({selector.url}).')
		
		if None == book_num_ratings:
			print(f'Warning: Book missing number of ratings ({selector.url}).')
		
		if None == book_num_reviews:
			print(f'Warning: Book missing number of reviews ({selector.url}).')
		
		if not len(book_genres):
			print(f'Warning: Book missing genres ({selector.url}).')
		
		if not len(book_description):
			print(f'Warning: Book missing description ({selector.url}).')
		
		book_df = pd.DataFrame()
		
		book_df['goodbooks_book_id'] = [goodbooks_book_id]
		book_df['title'] = [book_title]
		book_df['original_title'] = [book_orig_title]
		book_df['series'] = [book_series]
		book_df['language'] = [book_language]
		book_df['authors'] = ','.join(book_authors)
		book_df['avg_rating'] = book_avg_rating
		book_df['num_ratings'] = book_num_ratings
		book_df['num_reviews'] = book_num_reviews
		book_df['genres'] = ','.join(book_genres)
		book_df['description'] = book_description
		book_df['image_url'] = image_url
		
		book_df['url'] = selector.url 

		
		#SAVE EVERYTHING IN 1 DF
		global all_books_df
		all_books_df = pd.concat([all_books_df, book_df], sort=False)
	
	#Extracting the description of that particular book!!

	def process_book_description(self, spans):
		texts = []

		for desc_span in spans:
			desc_text = desc_span.strip()
			
			desc_text = desc_text.replace('\n', '')
			
			desc_text = desc_text.replace('\t', ' ')
			
			desc_text = re.sub(r"<[^>]*>", '', desc_text)

			if len(desc_text):
				texts.append(desc_text)
	
		return texts


#set the start and end page!!! as of now 1 page = 100 books
 
START_PAGE = 1
END_PAGE = 100

all_books_df = pd.DataFrame()

process = CrawlerProcess()
process.crawl(GoodReadsSpider)
process.start()

output_csv = f'C:/Users/kssan/OneDrive/Desktop/FINALPROJJ/data/book_original.csv'
all_books_df.to_csv(output_csv, index=False)


df=pd.read_csv(r"C:/Users/kssan/OneDrive/Desktop/FINALPROJJ/data/book_original.csv")

#Generating book_id!!
no_rows=len(df.index)
idx = 0
l=[]
   
for b_id in range(0,no_rows):  
    b_id=b_id+1
    l.append(b_id)
df.insert(loc=idx, column='book_id', value=l)
df.to_csv(r"C:\Users\kssan\OneDrive\Desktop\FINALPROJJ\data\books.csv", index=False)
