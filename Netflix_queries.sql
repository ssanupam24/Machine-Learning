Queries:
#######################################################################################################
Worst movies:
select count(1) as num1, mt.title, mt.yearofrelease from movie_titles mt JOIN movie_ratings mr ON
(mt.mid = mr.mid) AND (mr.rating BETWEEN 1 AND 2) AND
(mt.yearofrelease BETWEEN "1950" AND "2000")
GROUP BY mt.title, mt.yearofrelease
ORDER BY num1 DESC
limit 10;
#######################################################################################################
Good Movies:
select count(1) as num1, mt.title, mt.yearofrelease from movie_titles mt JOIN movie_ratings mr ON
(mt.mid = mr.mid) AND (mr.rating BETWEEN 4 AND 5) AND
(mt.yearofrelease BETWEEN "1950" AND "2000")
GROUP BY mt.title, mt.yearofrelease
ORDER BY num1 DESC
limit 10;
#######################################################################################################
100 Popular movies:

select title from movie_titles mt join 
(select mid from (select mid,count(customer_id) as c_custid from movie_ratings 
group by mid order by c_custid desc limit 100) a ) n 
where n.mid = mt.mid;
#######################################################################################################
Similar movies(general query):

select mr.mid,count(1) as count1 from movie_ratings mr left outer join
(select distinct mid,customer_id,rating from movie_ratings)a on a.customer_id = mr.customer_id AND
a.rating = mr.rating AND
a.mid is null
group by mr.mid,a.mid
order by count1 desc
limit 10; 
#######################################################################################################
New recommendation query:

select distinct mr1.mid,mr1.customer_id from movie_ratings mr1 left outer join
(select mr.customer_id,mr.mid,mr.rating from movie_ratings mr join 
(select mid from movie_ratings where customer_id = '716091' and rating = '5' limit 1 )a on mr.mid = a.mid and mr.rating = '5' and not mr.customer_id = '716091' limit 5)b on not mr1.customer_id = '716091' AND mr1.customer_id = b.customer_id AND
mr1.rating = '5' AND
b.mid is null
limit 10;
#######################################################################################################
Top 10 Most watched movies in 50 years

select count(mr.mid) as max1,mt.title,mt.yearofrelease from movie_ratings mr join movie_titles mt on
mt.mid=mr.mid AND
mt.yearofrelease BETWEEN "1950" AND "2000"
group by mt.title,mt.yearofrelease  
order by max1 desc
limit 10;	
#######################################################################################################
Top 10 average ratings of movies:

select mt.mid,avg(mr.rating) as average,mt.title,count(mr.mid) as count1 from movie_ratings mr join
movie_titles mt on mt.mid=mr.mid
group by mt.title,mt.mid
order by average desc
limit 10;
#######################################################################################################
Top 10 Active Viewer ratings of Netflix in 50 years.

select mr.customer_id,mr.rating,count(mr.rating) as count1 from movie_ratings mr join
(select customer_id,count(customer_id) as max1 from movie_ratings
where to_date(date) between '1950-01-01' AND '2000-12-31'
group by customer_id
order by max1 desc
limit 10)a on mr.customer_id = a.customer_id
group by mr.customer_id,mr.rating;	
#######################################################################################################
Top 10 no of 5 ratings by customer_id = '6':

select count(mr.customer_id) as max1,mt.yearofrelease from movie_ratings mr join movie_titles mt on 
mr.mid = mt.mid AND
mr.customer_id='6' AND
mt.yearofrelease BETWEEN "1950" AND "2000" 
group by mt.yearofrelease
order by max1 desc
limit 10;
#######################################################################################################
