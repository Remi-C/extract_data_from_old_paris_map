


--
-- load extracted numbers from jacoubet:  
DROP TABLE IF EXISTS annotations_detection ; 
CREATE TABLE annotations_detection (
gid serial
, image_name text
,x float -- expressed in lambert 1 carto
, y float
, orientations text
, detected_text text
) ; 

-- temporary table for the copy
DROP TABLE IF EXISTS annotations_detection_temp ; 
CREATE TABLE annotations_detection_temp (
content text
);


-- Feuille14_4000_4000_(681, 506, 730, 557);600574.91683774;1130611.5774229781;{10, 20, 80, 100, 170, 190, 200, 260, 280, 350};{"31", "31", "«à", "31", "7£7", "7£‘", ""ï", "7£7", "7£‘", "«à"}


-- read the csv file, but not as cv as it contains  not safe text from ocr
	COPY
		annotations_detection_temp 
	FROM 
	'/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/results/annotations_ocr/annotation_ocr.csv'
	 WITH CSV HEADER DELIMITER AS ';' ;


--cleaning the input to remove dangerous character such as quote
	WITH splitted_csv AS (
		 SELECT     f[1] AS image_name
			, f[2]::float AS x
			, f[3]::float AS y
			, f[4]::int[] AS orientations 
			, safe_text_9::text[] AS detected_text
		 FROM annotations_detection_temp
			, regexp_matches(content, E'(.+?);(.+?);(.+?);(.+?);(.*)$')as f --detect basic layout
			, replace(f[5], '$', '£') as safe_text_1 --we start the process of going from double quot to dollar quote
			, trim(both '|' from safe_text_1) AS safe_text_2 -- dont understand why there are these characters at all.
			, replace(safe_text_2,'", "', '$$, $$') as safe_text_3
			, regexp_replace(safe_text_3, '^{"', '{$$') AS safe_text_4
			, regexp_replace(safe_text_4, E'"}$', '$$}') AS safe_text_5
			--, regexp_matches(safe_text_3, E'^{"(.*)"}$')as safe_text_4
			--, concat('{$$', safe_text_4, '$$}') AS safe_text_5
			, replace(safe_text_5,'"', '``') as safe_text_6 
			, regexp_replace(safe_text_6,'''', '`') as safe_text_7 
			, replace(safe_text_7,'$$', '"') as safe_text_8 
			, replace(safe_text_8,'\', '/') as safe_text_9 --removing this very not safe character
		WHERE f[1] NOT ILIKE '%Feuille41_6000_0_(708, 1136, 751, 1174)%' --note : this row is corrupted !  
	)
	INSERT INTO annotations_detection (image_name, x, y , orientations, detected_text)
	SELECT *
	FROM splitted_csv ; 



	--now we need to select the actual digits that are going to  be used, among all the text detected.

	--simple ruel  : remove all non digit character, then take the number (if any) that is most likely

	SELECT gid,  te
	FROM annotations_detection
		, CAST(detected_text  AS text) AS te 
	WHERE image_name ILIKE '%Feuille23_2000_4000_(1275, 1282, 1318, 1332)%' ; 

	SELECT *
	FROM CAST '{32,22,22,32,22,22,€!“,ZZ,"{£",€!“,ZZ,"{£"}' AS text) as te
		, substring( te FROM '[0-9]+')
		, regexp_replace(te, '[^[{},:digit:]]','','g') 


	S


	
 
