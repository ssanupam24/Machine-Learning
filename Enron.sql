ENRON QUERIES

######################################################################################################


// To FIND NO of emails sent per year, per month

INSERT OVERWRITE LOCAL DIRECTORY 
'/home/hduser/enron2.csv' 
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
select month, year,count(*) AS count
from (select split(ts,' ')[2] AS month, split(ts,' ')[3] AS year from enron)E
group by month, year
order by count

######################################################################################################

//frequency of top 20 three word phrases used in enron dataset

INSERT OVERWRITE LOCAL DIRECTORY 
'/home/hduser/enron2.csv' 
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
 SELECT explode(ngrams(sentences(lower(context)), 3, 20)) AS x FROM Enron

######################################################################################################

// find no of emails sent by top 100 users

INSERT OVERWRITE LOCAL DIRECTORY 
'/home/hduser/enron2.csv' 
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
select sender, count(1) as count
FROM Enron
group by sender
having sender like'%enron.com'
order by count DESC 
limit 100;


######################################################################################################

// TO FIND LENGTH OF BODY OF EMAIL SENT BY EACH ENRON USER

INSERT OVERWRITE LOCAL DIRECTORY 
'/home/hduser/enron2.csv' 
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
select sender, sum(emailLength) as total
from (select sender, length(context) AS emailLength from enron) E 
group by sender 
having sender like '%enron.com'
order by total desc
limit 100;

######################################################################################################


// Finding 20 employees with most number of emails

INSERT OVERWRITE TABLE
senderData
select sender, count(*) AS count
FROM Enron
LATERAL VIEW 
explode(split(to, ',')) tempTable AS toParsed
group by sender
having sender like '%enron.com'
order by count desc
limit 20

// Creating Temprorary table to store all sender,receiver pairs

create table senderRec (sender STRING, Rec STRING) 
ROW FORMAT DELIMITED
STORED AS TEXTFILE;

//Finding all sender, Receiver pairs

INSERT OVERWRITE TABLE
senderRec
SELECT sender, toParsed
FROM Enron E
LATERAL VIEW 
explode(split(to, ',')) tempTable AS toParsed


//Creating Temprorary table to store sender,receiver pairs for the 20 employees found above

create table senderRecTop (sender STRING, Rec STRING) 
ROW FORMAT DELIMITED
STORED AS TEXTFILE;

//Storing sender,receiver pairs for the senders who are amongst the 20 employees found above

INSERT OVERWRITE TABLE
senderRecTop
select SenderRec.sender, Rec from senderRec join senderData on senderRec.sender = senderData.sender;

//storing sender and receiver amongst the 20 employee found above

INSERT OVERWRITE LOCAL DIRECTORY 
'/home/hduser/enron2.csv' 
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','

######################################################################################################
