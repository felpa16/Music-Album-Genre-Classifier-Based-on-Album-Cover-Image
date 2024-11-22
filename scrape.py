import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
import time
import re
import csv
import random
from collections import Counter

# Create folder to save album covers
os.makedirs('album_covers', exist_ok=True)

# Function to scrape album data
def scrape_album(album_id):

    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1"
    ]
    headers = {"User-Agent": random.choice(user_agents)}

    url = f'https://www.albumoftheyear.org/album/{album_id}/'
    response = requests.get(url, headers=headers)
    
    # Check if the page was found
    if response.status_code != 200:
        print(f"Album ID {album_id} not found, skipping...")
        return False  # Skip if album page does not exist
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get the album cover image URL
    img_tag = soup.find('link', rel='image_src')
    if img_tag:
        img_url = img_tag['href']
        
        # Extract genres for filename
        meta_description = soup.find('meta', attrs={'name': 'Description'})
        if meta_description:
            content = meta_description['content']
            # Use regular expression to find genre after "Genre: "
            genre_match = re.search(r'Genre:\s*([^\.\"]+)', content)
            genre = genre_match.group(1) if genre_match else 'Unknown'
        else:
            genre = 'Unknown'

        if genre == 'Unkown':
            return False

        # Fetch and save album cover with genre in filename
        img_response = requests.get(img_url)
        img = Image.open(BytesIO(img_response.content))
        img.save(f"album_covers/{genre}_{album_id}.jpg")

        print(f"Saved cover for Album ID {album_id} with genre: {genre}")
        return True
    else:
        print(f"Image not found for Album ID {album_id}, skipping...")
        return False

#6401 albumes incluyendo unknowns



# Loop through a range of album IDs
# successful_scrapes = 0
# for album_id in range(6001, 7000):  # Adjust range as needed
#     if scrape_album(album_id):
#         successful_scrapes += 1

#     # Break if we have reached 50,000 successful scrapes
#     if successful_scrapes >= 50000:
#         break

#     # Optional: Delay to avoid hitting the server too quickly
#     time.sleep(0.5)  # Adjust delay as necessary

# with open('successful_scrapes.csv', mode='a', newline='') as counter:
#     writer = csv.writer(counter)
#     writer.writerow([successful_scrapes])


def list_prefixes(directory, separator="_", unique=True):

    genres = []

    for filename in os.listdir(directory):
        prefix = filename.split(separator)[0]
        genres.append(prefix)

    if unique:
        genres = set(genres)

    return genres

# directory_path = "/Users/felipe/Vision/tp_final/album_covers"
# genres = list_prefixes(directory_path)

# # genre_string = "Comedy Rock, Heartland Rock, Pub Rock, Garage Rock, Blues Rock, Soft Rock, Glam Rock, Funk Rock, Surf Pop, Stoner Rock, Jangle Pop, Rock Opera, Rock & Roll, Country Rock, Gothic Rock, Southern Rock, Roots Rock, Piano Rock, Heavy Psych, Space Rock Revival, Christian Rock, Rock, Hard Rock, Progressive Rock, Post-Grunge, Power Pop, Groove Metal, New Wave, Punk Rock, Alternative Rock, Blues, Melodic Hardcore, Rockabilly, Post-Hardcore, Surf Punk, Jam Band, Merseybeat, Pigfuck, Alternative Dance, Indie Pop, Indie Rock, Alt-Pop, Post-Britpop, Britpop, Indie Surf, Garage Rock Revival, Dance-Punk, Grunge, Slacker Rock, Math Rock, Art Rock, Acoustic Rock, Alternative Metal, Nu Metal, Geek Rock, Experimental Rock, Punk, Post-Punk Revival, Post-Punk, Punk Rock, Pop Punk, Ska Punk, Hardcore Punk, Emoviolence, Riot Grrrl, Skate Punk, Crust Punk, Garage Punk, Punk Blues, Folk Punk, Dance-Punk, Art Punk, Midwest Emo, Post-Hardcore, Baggy, Emo, Rap Rock, Ska, Heavy Metal, Thrash Metal, Death Metal, Doom Metal, Black Metal, Progressive Metal, Power Metal, Grindcore, Mathcore, Metalcore, Sludge Metal, Gothic Metal, Rap Metal, Melodic Black Metal, Brutal Death Metal, Symphonic Black Metal, Melodic Death Metal, Doomgaze, Brutal Prog, Pagan Metal, Black 'n' Roll, Progressive Metalcore, Atmospheric Black Metal, Deathcore, Stoner Metal, Drone Metal, Neue Deutsche Härte, Melodic Metalcore, Country Rock, Folk Rock, Psychedelic Rock, Jazz-Rock, Blues Rock, Funk Metal, Punk Blues, Garage Rock Revival, Glam Rock, Heavy Psych, Krautrock, Canterbury Scene, Jazz Fusion, Blues, Southern Rock, Latin Rock, Post-Metal, Progressive Rock, Symphonic Prog, Progressive Pop, House, Techno, Acid House, Deep House, Drum and Bass, Big Beat, Electro House, Trap (EDM), Complextro, Nu-Disco, Progressive House, Dubstep, Future Bass, Dancehall, Electronic Dance Music, Footwork, Breakbeat Hardcore, Electroclash, Electro Swing, Bitpop, Tech House, Microhouse, Juke, Brostep, Chillwave, New Rave, Synthwave, Synthpop, Minimal Wave, Trap, Vapor, Future Garage, Electropop, UK Garage, Balearic Beat, Synthwave, Chillwave, Minimal Wave, Electroclash, Synthpop, Synth Funk, Electro Swing, Electro-Disco, French House, New Rave, Italo-Disco, Outsider House, Minimal Synth, Hip House, Hypnagogic Pop, Indietronica, Witch House, Futurepop, Electropop, Industrial, Electro-Industrial, Noise Rock, Post-Industrial, Industrial Rock, Digital Hardcore, Industrial Metal, Industrial Techno, Industrial Hip Hop, IDM, Deconstructed Club, Glitch, Ambient Techno, Dark Ambient, Vaporwave, Wonky, Glitch Hop, Progressive Electronic, Drill and Bass, Atmospheric Drum and Bass, Ambient Dub, Ambient, New Age, Chillstep, Ambient Pop, Chillwave, Minimal Techno, Dub Techno, Hauntology, Reductionism, Downtempo, Dub, Breakcore, Acid Techno, Darkwave, Electronic, Teen Pop, Alt-Pop, Pop Rock, Pop Punk, Dance-Pop, Sunshine Pop, Sophisti-Pop, Boy Band, French Pop, Pop Reggae, Pop Rap, Dance-Pop, Indie Pop, Math Pop, Latin Pop, Sophisti-Pop, Pop, R&B, Pop Soul, Contemporary R&B, Neo-Soul, Blue-Eyed Soul, Smooth Soul, Jazz-Funk, Southern Soul, Deep Soul, Psychedelic Soul, Alternative R&B, Soul, Funk, Funk Rock, Funky House, Gospel, New Orleans R&B, Progressive Soul, Indie Pop, Twee Pop, Chamber Pop, Baroque Pop, Folk Pop, Bedroom Pop, Jangle Pop, Chamber Pop, Sunshine Pop, Alt-Pop, Art Pop, Chamber Pop, Synthpop, Sophisti-Pop, Indie Pop, Psychedelic Pop, Vocal Jazz, Cool Jazz, Jazz Fusion, Jazz-Rock, Nu Jazz, Acid Jazz, Jazz Rap, Ethio-Jazz, Dixieland, Chamber Jazz, Third Stream, ECM Style Jazz, Afro-Cuban Jazz, Acid Jazz, Jazz Fusion, Chamber Jazz, Jazzstep, Soul Jazz, Avant-Garde Jazz, Jazz Pop, Jazz, Jazz Fusion, Jazz-Funk, ECM Style Jazz, Afro-Cuban Jazz, Acid Jazz, Cool Jazz, Jazz Rap, Neo-Soul, Funk Metal, Acid Jazz, Hip Hop, Southern Hip Hop, East Coast Hip Hop, West Coast Hip Hop, Conscious Hip Hop, Gangsta Rap, Boom Bap, Jazz Rap, G-Funk, Abstract Hip Hop, Hardcore Hip Hop, Memphis Rap, Comedy Rap, Instrumental Hip Hop, Glitch Hop, Jazz Rap, Chipmunk Soul, Cloud Rap, Southern Hip Hop, UK Hip Hop, Hardcore Hip Hop, French Hip Hop, Horrorcore, Trip Hop, Trap (EDM), Drill and Bass, Cloud Rap, Wonky, Vaporwave, UK Bass, Footwork, Turntablism, Industrial Hip Hop, American Primitivism, Avant-Folk, Folk Punk, Contemporary Folk, Indie Folk, Folk Metal, Chamber Folk, Americana, Progressive Folk, Progressive Bluegrass, American Folk Music, Chamber Folk, Psychedelic Folk, Blues, Folk, Singer-Songwriter, Folk, Roots Reggae, Country, Alt-Country, Country Pop, Country Rock, Outlaw Country, Progressive Country, Americana, Gothic Country, Country Blues, Bluegrass, Contemporary Country, Country Rock, Sound Collage, Free Improvisation, Avant-Garde Metal, Avant-Folk, Experimental Hip Hop, Experimental Rock, Drone, Minimalism, Outsider House, Post-Industrial, Hauntology, Moogsploitation, Freak Folk, Psichedelia occulta italiana, Reductionism, Modern Classical, Experimental, Noise, Dark Ambient, Ambient, Drone, Noise Pop, Minimal Synth, Ambient Dub, Dark Ambient, Ambient Techno, Ambient Dub, Noise Rock, Witch House, Drone Metal, Slowcore, Dark Cabaret, Ethereal Wave, Atmospheric Sludge Metal, Folktronica, Musique concrète, Plunderphonics, Sampledelia, Tape Music, Glitch Pop, Moogsploitation, Mashup, Turntable Music, Contemporary Classical, Minimalism, Post-Minimalism, Third Stream, Cinematic Classical, Chamber Folk, Standards, Opera, Modern Classical, Neoclassical Darkwave, Afrobeat, Ethio-Jazz, Tishoumaren, Cumbia, Latin Rock, Latin Alternative, Mande Music, MPB, Afro-Cuban Jazz, Gospel, American Folk Music, Celtic Punk, Gothic Country, Bluegrass, Psychedelia, Comedy, Stand-Up Comedy, Christmas, Unknown, Spoken Word, Shitgaze"
# genre_string = "Comedy Rock, Heartland Rock, Pub Rock, Garage Rock, Blues Rock, Soft Rock, Glam Rock, Funk Rock, Surf Pop, Stoner Rock, Jangle Pop, Rock Opera, Rock & Roll, Country Rock, Gothic Rock, Southern Rock, Roots Rock, Piano Rock, Heavy Psych, Space Rock Revival, Christian Rock, Rock, Hard Rock, Progressive Rock, Post-Grunge, Power Pop, Groove Metal, New Wave, Punk Rock, Alternative Rock, Blues, Melodic Hardcore, Rockabilly, Post-Hardcore, Surf Punk, Jam Band, Merseybeat, Electric Blues, Alternative Dance, Indie Pop, Indie Rock, Alt-Pop, Post-Britpop, Britpop, Indie Surf, Garage Rock Revival, Dance-Punk, Grunge, Slacker Rock, Math Rock, Art Rock, Acoustic Rock, Alternative Metal, Nu Metal, Geek Rock, Experimental Rock, Punk, Post-Punk Revival, Post-Punk, Shoegaze, Psychedelia, Neo-Psychedelia, Post-Rock, Punk Rock, Pop Punk, Ska Punk, Hardcore Punk, Emoviolence, Riot Grrrl, Skate Punk, Crust Punk, Garage Punk, Punk Blues, Folk Punk, Dance-Punk, Art Punk, Midwest Emo, Post-Hardcore, Baggy, Emo, Rap Rock, Ska, Heavy Metal, Thrash Metal, Death Metal, Doom Metal, Black Metal, Progressive Metal, Power Metal, Grindcore, Mathcore, Metalcore, Sludge Metal, Gothic Metal, Rap Metal, Melodic Black Metal, Brutal Death Metal, Symphonic Black Metal, Melodic Death Metal, Doomgaze, Brutal Prog, Pagan Metal, Black 'n' Roll, Progressive Metalcore, Atmospheric Black Metal, Deathcore, Stoner Metal, Drone Metal, Neue Deutsche Härte, Melodic Metalcore, Blackgaze, Country Rock, Folk Rock, Psychedelic Rock, Jazz-Rock, Blues Rock, Funk Metal, Punk Blues, Garage Rock Revival, Glam Rock, Heavy Psych, Krautrock, Canterbury Scene, Jazz Fusion, Blues, Southern Rock, Latin Rock, Post-Metal, Progressive Rock, Symphonic Prog, Progressive Pop, House, Techno, Acid House, Deep House, Drum and Bass, Big Beat, Electro House, Trap (EDM), Complextro, Nu-Disco, Progressive House, Dubstep, Future Bass, Dancehall, Electronic Dance Music, Footwork, Breakbeat Hardcore, Electroclash, Electro Swing, Bitpop, Tech House, Microhouse, Juke, Brostep, Chillwave, New Rave, Synthwave, Synthpop, Minimal Wave, Trap, Vapor, Future Garage, Electropop, UK Garage, Balearic Beat, Electronic, Electro, Synthwave, Chillwave, Minimal Wave, Electroclash, Synthpop, Synth Funk, Electro Swing, Electro-Disco, French House, New Rave, Italo-Disco, Outsider House, Minimal Synth, Hip House, Hypnagogic Pop, Indietronica, Witch House, Futurepop, Electropop, Industrial, Electro-Industrial, Noise Rock, Post-Industrial, Industrial Rock, Digital Hardcore, Industrial Metal, Industrial Techno, Industrial Hip Hop, IDM, Deconstructed Club, Glitch, Ambient Techno, Dark Ambient, Vaporwave, Wonky, Glitch Hop, Progressive Electronic, Drill and Bass, Atmospheric Drum and Bass, Ambient Dub, Ambient, New Age, Chillstep, Ambient Pop, Chillwave, Minimal Techno, Dub Techno, Hauntology, Reductionism, Downtempo, Dub, Breakcore, Acid Techno, Darkwave, Teen Pop, Alt-Pop, Pop Rock, Pop Punk, Dance-Pop, Sunshine Pop, Sophisti-Pop, Boy Band, French Pop, Pop Reggae, Pop Rap, Dance-Pop, Indie Pop, Math Pop, Latin Pop, Sophisti-Pop, Pop, Disco, R&B, Pop Soul, Contemporary R&B, Neo-Soul, Blue-Eyed Soul, Smooth Soul, Jazz-Funk, Southern Soul, Deep Soul, Psychedelic Soul, Alternative R&B, Soul, Funk, Funk Rock, Funky House, Gospel, New Orleans R&B, Progressive Soul, Indie Pop, Twee Pop, Chamber Pop, Baroque Pop, Folk Pop, Bedroom Pop, Jangle Pop, Chamber Pop, Sunshine Pop, Alt-Pop, Art Pop, Chamber Pop, Synthpop, Sophisti-Pop, Indie Pop, Psychedelic Pop, Dream Pop, Vocal Jazz, Cool Jazz, Jazz Fusion, Jazz-Rock, Nu Jazz, Acid Jazz, Jazz Rap, Ethio-Jazz, Dixieland, Chamber Jazz, Third Stream, ECM Style Jazz, Afro-Cuban Jazz, Acid Jazz, Jazz Fusion, Chamber Jazz, Jazzstep, Avant-Garde Jazz, Jazz Pop, Jazz, Jazz Fusion, Jazz-Funk, ECM Style Jazz, Afro-Cuban Jazz, Acid Jazz, Cool Jazz, Jazz Rap, Neo-Soul, Funk Metal, Acid Jazz, Hip Hop, Southern Hip Hop, East Coast Hip Hop, West Coast Hip Hop, Conscious Hip Hop, Gangsta Rap, Boom Bap, Jazz Rap, G-Funk, Abstract Hip Hop, Hardcore Hip Hop, Memphis Rap, Comedy Rap, Instrumental Hip Hop, Glitch Hop, Jazz Rap, Chipmunk Soul, Cloud Rap, Southern Hip Hop, UK Hip Hop, Hardcore Hip Hop, French Hip Hop, Horrorcore, Trip Hop, Trap [EDM], Drill and Bass, Cloud Rap, Wonky, Vaporwave, UK Bass, Footwork, Turntablism, Industrial Hip Hop, American Primitivism, Avant-Folk, Folk Punk, Contemporary Folk, Indie Folk, Folk Metal, Chamber Folk, Americana, Progressive Folk, Progressive Bluegrass, American Folk Music, Chamber Folk, Psychedelic Folk, Blues, Folk, Singer-Songwriter, Folk, Roots Reggae, Country, Alt-Country, Country Pop, Country Rock, Outlaw Country, Progressive Country, Americana, Gothic Country, Country Blues, Bluegrass, Contemporary Country, Country Rock, Sound Collage, Free Improvisation, Avant-Garde Metal, Avant-Folk, Experimental Hip Hop, Experimental Rock, Drone, Minimalism, Outsider House, Post-Industrial, Hauntology, Moogsploitation, Freak Folk, Psichedelia occulta italiana, Reductionism, Modern Classical, Experimental, Noise, Dark Ambient, Ambient, Drone, Noise Pop, Minimal Synth, Ambient Dub, Dark Ambient, Ambient Techno, Ambient Dub, Noise Rock, Witch House, Drone Metal, Slowcore, Dark Cabaret, Ethereal Wave, Atmospheric Sludge Metal, Folktronica, Musique concrète, Plunderphonics, Sampledelia, Tape Music, Glitch Pop, Moogsploitation, Mashup, Turntable Music, Minimalism, Post-Minimalism, Third Stream, Cinematic Classical, Chamber Folk, Standards, Opera, Modern Classical, Neoclassical Darkwave, Afrobeat, Ethio-Jazz, Tishoumaren, Cumbia, Latin Rock, Latin Alternative, Mande Music, MPB, Afro-Cuban Jazz, Reggae, American Folk Music, Celtic Punk, Gothic Country, Bluegrass, Comedy, Stand-Up Comedy, Christmas, Unknown, Spoken Word, Shitgaze, Pigfuck"

# counter = 0
# genre_list1 = [i.lstrip() for i in genre_string.split(",")]

# genre_set = set()
# unique_genre_list = []

# for genre in genre_list1:
#     if genre not in genre_set:
#         unique_genre_list.append(genre)
#         genre_set.add(genre)

# unique_genre_string = ', '.join(unique_genre_list)
# print(unique_genre_list)

# genre_counts = Counter(genre_list1)
# duplicates = [image for image, count in genre_counts.items() if count > 1]
# print(duplicates)

# print(len(genre_list1))
# genre_list2 = list(set([i.lstrip() for i in string.split(",")]))
# print(len(genre_list2))

# for genre in genre_list2:
#     if genre.lstrip() in genres:
#         counter += 1

# counter = Counter(genres)
# duplicates = [item for item, count in counter.items() if count > 1]
# print(duplicates)

# print(genres - set(new_genre_list))
# print(len(genres - set(new_genre_list)))
# print('\n\n\n\n')
# print(set(genre_list2) - genres)

def sort_albums(separator='_'):
    
    path = "/Users/felipe/Vision/tp_final/album_covers"
    label_csv = "/Users/felipe/Vision/tp_final/labels.csv"

    labels = []

    rock = set(['Comedy Rock', 'Heartland Rock', 'Pub Rock', 'Garage Rock', 'Blues Rock', 'Soft Rock', 'Glam Rock', 'Funk Rock', 'Surf Pop', 'Stoner Rock', 'Jangle Pop', 'Rock Opera', 'Rock & Roll', 'Gothic Rock', 'Southern Rock', 'Roots Rock', 'Piano Rock', 'Heavy Psych', 'Space Rock Revival', 'Christian Rock', 'Rock', 'Hard Rock', 'Progressive Rock', 'Post-Grunge', 'Power Pop', 'Groove Metal', 'New Wave', 'Alternative Rock', 'Blues', 'Melodic Hardcore', 'Rockabilly', 'Post-Hardcore', 'Surf Punk', 'Jam Band', 'Merseybeat', 'Electric Blues', 'Folk Rock', 'Psychedelic Rock', 'Jazz-Rock', 'Krautrock', 'Canterbury Scene', 'Latin Rock', 'Symphonic Prog'])
    alt_rock = set(['Alternative Dance', 'Indie Rock', 'Alt-Pop', 'Post-Britpop', 'Britpop', 'Indie Surf', 'Garage Rock Revival', 'Dance-Punk', 'Grunge', 'Slacker Rock', 'Math Rock', 'Art Rock', 'Acoustic Rock', 'Alternative Metal', 'Nu Metal', 'Geek Rock', 'Experimental Rock', 'Punk', 'Post-Punk Revival', 'Post-Punk', 'Shoegaze', 'Psychedelia', 'Neo-Psychedelia', 'Post-Rock', 'Dream Pop', 'Noise Rock', 'Industrial Rock'])
    punk = set(['Punk Rock', 'Pop Punk', 'Ska Punk', 'Hardcore Punk', 'Emoviolence', 'Riot Grrrl', 'Skate Punk', 'Crust Punk', 'Garage Punk', 'Punk Blues', 'Folk Punk', 'Art Punk', 'Midwest Emo', 'Baggy', 'Emo', 'Rap Rock', 'Celtic Punk'])
    metal = set(['Heavy Metal', 'Thrash Metal', 'Death Metal', 'Doom Metal', 'Black Metal', 'Progressive Metal', 'Power Metal', 'Grindcore', 'Mathcore', 'Metalcore', 'Sludge Metal', 'Gothic Metal', 'Rap Metal', 'Melodic Black Metal', 'Brutal Death Metal', 'Symphonic Black Metal', 'Melodic Death Metal', 'Doomgaze', 'Brutal Prog', 'Pagan Metal', "Black 'n' Roll", 'Progressive Metalcore', 'Atmospheric Black Metal', 'Deathcore', 'Stoner Metal', 'Drone Metal', 'Neue Deutsche Härte', 'Melodic Metalcore', 'Blackgaze', 'Funk Metal', 'Post-Metal', 'Folk Metal', 'Avant-Garde Metal', 'Industrial Metal'])
    edm = set(['House', 'Techno', 'Acid House', 'Deep House', 'Drum and Bass', 'Big Beat', 'Electro House', 'Trap (EDM)', 'Complextro', 'Nu-Disco', 'Progressive House', 'Dubstep', 'Future Bass', 'Dancehall', 'Electronic Dance Music', 'Footwork', 'Breakbeat Hardcore', 'Electroclash', 'Electro Swing', 'Bitpop', 'Tech House', 'Microhouse', 'Juke', 'Brostep', 'Trap', 'Vapor', 'Future Garage', 'Electropop', 'UK Garage', 'Balearic Beat', 'Electronic', 'Electro', 'Industrial Techno'])
    synth_based = set(['Synth Funk', 'Electro-Disco', 'French House', 'Italo-Disco', 'Outsider House', 'Minimal Synth', 'Hip House', 'Hypnagogic Pop', 'Indietronica', 'Witch House', 'Futurepop', 'Chillwave', 'New Rave', 'Synthwave', 'Synthpop', 'Minimal Wave'])
    pop = set(['Teen Pop', 'Pop Rock', 'Dance-Pop', 'Sunshine Pop', 'Sophisti-Pop', 'Boy Band', 'French Pop', 'Pop Reggae', 'Pop Rap', 'Math Pop', 'Latin Pop', 'Pop', 'Disco'])
    soul_r_and_b = set(['R&B', 'Pop Soul', 'Contemporary R&B', 'Neo-Soul', 'Blue-Eyed Soul', 'Smooth Soul', 'Jazz-Funk', 'Southern Soul', 'Deep Soul', 'Psychedelic Soul', 'Alternative R&B', 'Soul', 'Funk', 'Funky House', 'Gospel', 'New Orleans R&B', 'Progressive Soul'])
    alt_pop = set(['Indie Pop', 'Twee Pop', 'Chamber Pop', 'Baroque Pop', 'Folk Pop', 'Bedroom Pop', 'Art Pop', 'Psychedelic Pop', 'Progressive Pop'])
    jazz = set(['Vocal Jazz', 'Cool Jazz', 'Nu Jazz', 'Acid Jazz', 'Jazz Rap', 'Ethio-Jazz', 'Dixieland', 'Chamber Jazz', 'Third Stream', 'ECM Style Jazz', 'Afro-Cuban Jazz', 'Jazzstep', 'Avant-Garde Jazz', 'Jazz Pop', 'Jazz', 'Jazz Fusion'])
    hip_hop_and_rap = set(['Hip Hop', 'Southern Hip Hop', 'East Coast Hip Hop', 'West Coast Hip Hop', 'Conscious Hip Hop', 'Gangsta Rap', 'Boom Bap', 'G-Funk', 'Abstract Hip Hop', 'Hardcore Hip Hop', 'Memphis Rap', 'Comedy Rap', 'Instrumental Hip Hop', 'Chipmunk Soul', 'Cloud Rap', 'UK Hip Hop', 'French Hip Hop', 'Horrorcore', 'Trip Hop', 'Trap [EDM]', 'UK Bass', 'Turntablism', 'Industrial Hip Hop'])
    folk = set(['American Primitivism', 'Avant-Folk', 'Contemporary Folk', 'Indie Folk', 'Chamber Folk', 'Americana', 'Progressive Folk', 'Progressive Bluegrass', 'American Folk Music', 'Psychedelic Folk', 'Folk', 'Singer-Songwriter', 'Roots Reggae', 'Freak Folk'])
    country = set(['Country Rock', 'Country', 'Alt-Country', 'Country Pop', 'Outlaw Country', 'Progressive Country', 'Gothic Country', 'Country Blues', 'Bluegrass', 'Contemporary Country'])
    experimental = set(['Sound Collage', 'Free Improvisation', 'Experimental Hip Hop', 'Drone', 'Minimalism', 'Moogsploitation', 'Psichedelia occulta italiana', 'Experimental', 'IDM', 'Deconstructed Club', 'Glitch', 'Vaporwave', 'Wonky', 'Glitch Hop', 'Progressive Electronic', 'Drill and Bass', 'Atmospheric Drum and Bass', 'New Age', 'Chillstep', 'Minimal Techno', 'Dub Techno', 'Hauntology', 'Reductionism', 'Downtempo', 'Dub', 'Breakcore', 'Acid Techno', 'Darkwave', 'Musique concrète', 'Plunderphonics', 'Sampledelia', 'Tape Music', 'Glitch Pop', 'Mashup', 'Turntable Music', 'Industrial', 'Electro-Industrial', 'Post-Industrial'])
    noise_and_ambient = set(['Noise', 'Noise Pop', 'Slowcore', 'Dark Cabaret', 'Ethereal Wave', 'Atmospheric Sludge Metal', 'Folktronica', 'Ambient Techno', 'Dark Ambient', 'Ambient Dub', 'Ambient', 'Ambient Pop'])
    world_music = set(['Afrobeat', 'Tishoumaren', 'Cumbia', 'Latin Alternative', 'Mande Music', 'MPB', 'Reggae', 'Ska'])

    discarded = set(['Comedy', 'Stand-Up Comedy', 'Christmas', 'Unknown', 'Spoken Word', 'Shitgaze', 'Pigfuck', 'Digital Hardcore', 'Modern Classical', 'Post-Minimalism', 'Cinematic Classical', 'Standards', 'Opera', 'Neoclassical Darkwave'])
    counter = 0
    
    with open(label_csv, mode='w', newline="") as file:
        writer = csv.writer(file)
        for image in os.listdir(path):
            prefix = image.split(separator)[0]
            counter += 1

            if prefix in rock:
                writer.writerow([image, 0])
            elif prefix in alt_rock:
                writer.writerow([image, 1])
            elif prefix in punk:
                writer.writerow([image, 2])
            elif prefix in metal:
                writer.writerow([image, 3])
            elif prefix in edm:
                writer.writerow([image, 4])
            elif prefix in synth_based:
                writer.writerow([image, 5])
            elif prefix in pop:
                writer.writerow([image, 6])
            elif prefix in soul_r_and_b:
                writer.writerow([image, 7])
            elif prefix in alt_pop:
                writer.writerow([image, 8])
            elif prefix in jazz:
                writer.writerow([image, 9])
            elif prefix in hip_hop_and_rap:
                writer.writerow([image, 10])
            elif prefix in folk:
                writer.writerow([image, 11])
            elif prefix in country:
                writer.writerow([image, 12])
            elif prefix in experimental:
                writer.writerow([image, 13])
            elif prefix in noise_and_ambient:
                writer.writerow([image, 14])
            elif prefix in world_music:
                writer.writerow([image, 15])
            else:
                path_to_image = os.path.join(path, image)
                os.remove(path_to_image)
                continue

    print(f"Total amount of labeled albums: {counter}")

sort_albums()