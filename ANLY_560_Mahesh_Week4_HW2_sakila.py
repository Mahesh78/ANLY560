"""
Using the Sakila database, create a query that displays 
the film title and description, as well the actor first and last name 
for all films that begin with the letters 'zo'. 

@author: Mahesh
"""

import pymysql
db = pymysql.connect("localhost","root","harris","sakila")
cursor = db.cursor()
cursor.execute('''SELECT a.title, a.description, c.first_name, c.last_name \
               FROM sakila.film a \
               LEFT JOIN sakila.film_actor b \
               ON a.film_id = b.film_id \
               LEFT JOIN sakila.actor c \
               ON c.actor_id = b.actor_id \
               WHERE a.title LIKE 'ZO%' ''')
result = cursor.fetchall()
print(result)