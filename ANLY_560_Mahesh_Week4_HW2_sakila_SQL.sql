SELECT a.title, a.description, c.first_name, c.last_name
FROM sakila.film a
LEFT JOIN sakila.film_actor b
ON a.film_id = b.film_id
LEFT JOIN sakila.actor c
ON c.actor_id = b.actor_id
WHERE a.title LIKE 'ZO%'