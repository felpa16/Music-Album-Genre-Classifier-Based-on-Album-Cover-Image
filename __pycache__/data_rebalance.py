import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import csv 
import torch

# Load the data
def sort_albums(separator='_'):
    
    path = "/home/olivia/Desktop/Vision_Artificial/TP_Final/album_covers"
    label_csv = "/home/olivia/Desktop/Vision_Artificial/TP_Final/labels.csv"

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

def class_distribution(filename, prints=False):
    df = pd.read_csv(filename, header=None)
    df.columns = ['image', 'label']
    df = df.groupby('label').count()
    df.reset_index(inplace=True)
    df.columns = ['label', 'count']
    if prints:
        print(df)
    return df

def plot_distribution(df):
    plt.bar(df['label'], df['count'])
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.title('Genre Distribution')
    plt.show()


df = class_distribution("/home/olivia/Desktop/Vision_Artificial/TP_Final/labels.csv")
plot_distribution(df)
print("Original dataset:")
print(df)

"""
Alternativas de imbalance de clases

1. Scrappear más datos (excluyendo generos con mayor representacion) y hacer undersampling
2. Hacer oversampling (cuidado con esto), que puede ser duplicar imagenes en el dataset de train o hacer smote
3. Ajustar pesos en la funcion de costo para equiparar las clases (no se como hacerlo en pytorch)
4. Fusionar o eliminar clases del dataset (Esto cambia el problema)
5. Elegir un numero n. La idea es scrappear mas muestras para las clases con menos de n muestras y hacer undersampling para las clases con mas de n muestras.
"""


def undersample(min_samples, original, newfile):
    df = pd.read_csv(original, header=None)
    df.columns = ['image', 'label']
    df = df.groupby('label').apply(lambda x: x.sample(min_samples))
    df.reset_index(drop=True, inplace=True)
    df.to_csv(newfile, header=False, index=False)
    print(f"Undersampled dataset saved to {newfile}")
    print("New dataset:")
    print(df)
    return df


#undersample(df['count'].min(), "/home/olivia/Desktop/Vision_Artificial/TP_Final/labels.csv", "/home/olivia/Desktop/Vision_Artificial/TP_Final/labels_undersample.csv")
#df_under = class_distribution("/home/olivia/Desktop/Vision_Artificial/TP_Final/labels_undersample.csv", True)
#plot_distribution(df_under)


def weighted_sampling(df, newfile):
    weights = 1 / df['count']
    weights = weights / weights.sum()
    df['weights'] = weights
    print(df)
    weights_ = weights.to_dict()
    df.to_csv(newfile, header=False, index=False)
    return weights_

#df_weighted_dict = weighted_sampling(df, "/home/olivia/Desktop/Vision_Artificial/TP_Final/labels_weighted.csv")

def weighted_cross_entropy(output, target, weights):
    """
    output: tensor of shape (batch_size, num_classes)
    target: tensor of shape (batch_size)
    weights: dictionary of class weights
    """
    ce = -weights[target] * target * torch.log(output)
    return ce.mean()

def merge_classes(counts, original, newfile):
    average = sum(counts['count']) // len(counts['count'])
    counts = counts[counts['count'] > average]
    counts = counts['label'].tolist()
    

    df = pd.read_csv(original, header=None)
    df.columns = ['image', 'label']
    df['label'] = df['label'].apply(lambda x: x if x not in counts else 'merged')

average = sum(df['count']) // len(df['count'])
df_merged = merge_classes(average, "/home/olivia/Desktop/Vision_Artificial/TP_Final/labels.csv", "/home/olivia/Desktop/Vision_Artificial/TP_Final/labels_merged.csv")