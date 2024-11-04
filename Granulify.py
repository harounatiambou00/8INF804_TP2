from typing import List, Tuple
import numpy as np
import pandas as pd
import cv2
from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import KMeans
import os


class Granulify:
    def __init__(self, images: List[Tuple[np.array, str]], n_clusters: int = 20, slic_segments: int = 300, display: bool = False):
        self.n_clusters = n_clusters
        self.slic_segments = slic_segments
        self.display = display
        self.images = images

        self.output_folder = "./Images"
        os.makedirs(self.output_folder, exist_ok=True)

    def __preprocess(self, image: np.array) -> np.array:
        # Affiche l'image originale si self.display est activé
        if self.display:
            cv2.imshow('Image originale', image)
            cv2.waitKey(0)

        # Conversion en LAB pour appliquer CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self.display:
            cv2.imshow('Apres CLAHE', img_clahe)
            cv2.waitKey(0)

        # Réduction de bruit avec un filtre bilatéral
        img_denoised = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

        if self.display:
            cv2.imshow('Apres Bilateral Filter', img_denoised)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return img_denoised

    def __segment(self, image: np.array) -> np.array:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_preprocessed = self.__preprocess(image)

        if self.display:
            cv2.imshow('Image pretraitee', image_preprocessed)
            cv2.waitKey(0)

        # Extraction de l'arrière-plan en appliquant un seuil
        _, background_mask = cv2.threshold(gray_image, 50, 1, cv2.THRESH_BINARY)

        if self.display:
            cv2.imshow('Masque d arrière plan', background_mask * 255)
            cv2.waitKey(0)

        # Conversion de l'image en espace de couleur XYZ
        image_xyz = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

        # Nettoyage de l'image avec le masque d'arrière-plan
        cleaned_image = cv2.bitwise_and(image, image, mask=background_mask)
        blurred_image = cv2.GaussianBlur(cleaned_image, (101, 101), 0)

        # Conversion en espaces de couleur LAB et HSV
        blurred_image_lab = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2LAB)
        blurred_image_hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        # Clustering des pixels en utilisant KMeans
        mask = background_mask != 0
        pixel_features = np.concatenate((blurred_image_lab, blurred_image_hsv), axis=2)[mask, :]
        labels = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto').fit_predict(pixel_features)

        # Reshape des labels pour correspondre à la forme de l'image
        clustered_image = np.zeros_like(blurred_image_lab[:, :, 1])
        c = 0
        for i in range(clustered_image.shape[0]):
            for j in range(clustered_image.shape[1]):
                if mask[i, j]:
                    clustered_image[i, j] = labels[c]
                    c += 1

        if self.display:
            cv2.imshow('Clustering avec KMeans', (clustered_image / clustered_image.max() * 255).astype(np.uint8))
            cv2.waitKey(0)

        # Application de la segmentation SLIC
        slic_segmentation = slic(image_xyz, n_segments=self.slic_segments, compactness=20, sigma=1)
        slic_segmentation = slic_segmentation * background_mask

        # Nettoyage de la segmentation SLIC pour éliminer les petits superpixels
        for i in range(1, slic_segmentation.max() + 1):
            if np.count_nonzero(slic_segmentation == i) < 500:
                slic_segmentation[slic_segmentation == i] = 0

        segmentation_with_boundaries = mark_boundaries(image, slic_segmentation)

        if self.display:
            cv2.imshow('Segmentation Finale avec Contours', (segmentation_with_boundaries * 255).astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return slic_segmentation

    def __extract_features(self, superpixel: np.array) -> np.array:
        mean_bgr = np.mean(superpixel, axis=(0, 1))
        std_bgr = np.std(superpixel, axis=(0, 1))

        # Combinaison des caractéristiques en un tableau 1D
        features = np.concatenate([mean_bgr, std_bgr])

        # Gestion des valeurs NaN (cas où le superpixel n'a pas de pixels valides)
        features = np.nan_to_num(features, nan=0.0)

        return features

    def __save_results(self, segmentation: np.array, image: np.array, filename: str) -> None:
        results = []

        for label in range(1, segmentation.max() + 1):
            superpixel_mask = (segmentation == label)

            if np.count_nonzero(superpixel_mask) > 0:
                superpixel = image * superpixel_mask[..., np.newaxis]
                features = self.__extract_features(superpixel)
                results.append([label] + features.tolist())

        columns = ["Superpixel"] + ["Moyenne de B", "Moyenne de G", "Moyenne de R", "Écart type de B",
                                    "Écart type de G", "Écart type de R"]
        results_df = pd.DataFrame(results, columns=columns)

        csv_filename = os.path.join(self.output_folder, filename.split('.')[0] + '_segmentation_results.csv')
        results_df.to_csv(csv_filename, index=False)

        segmented_with_boundaries = mark_boundaries(image, segmentation)
        segmented_filename = os.path.join(self.output_folder, filename.split('.')[0] + '_segmented_with_boundaries.png')
        cv2.imwrite(segmented_filename, (segmented_with_boundaries * 255).astype(np.uint8))

        print(f"Les résultats de la segmentation de {filename} ont été sauvegardés dans le fichier : {csv_filename}")

    def run(self) -> None:
        for image, filename in self.images:
            if image is None:
                print(f"Erreur : {filename} est corrompue. Image ignorée.")
                continue
            print(f"Traitement de l'image : {filename}")
            segmentation = self.__segment(image)
            self.__save_results(segmentation, image, filename)
        print("Traitement terminé pour toutes les images.")
