# ProjetLoto
Reconnaissance de nombres contenus sur surfaces sphériques : application au tirage du loto

===

Le fichier "fichier_train_generique.py" (celui permettant de créer un CNN et de l'entraîner) se trouve dans le dossier : <strong>RenduStage</strong>
  
Il n'y a pas ici de jeu de données fourni (propriété de l'entreprise avec laquelle j'ai effectué mon stage), mais l'architecture du dossier doit être la suivante :<br>
&nbsp&nbsp&nbsp&nbsp- JeuDeDonnee (le nom de ce dossier n'a pas d'importance)<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp- Boule1<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp- 1_num_1.jpg<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp- 1_num_2.jpg<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp- ...<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp- Boule 2<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp-  2_num1.jpg<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp-  2_num2.jpg<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp-  ...<br>
         
    

Le fichier "prediction.py" (celui permettant d'utiliser le programme, une fois un CNN entraîné) se trouve dans le dossier : <strong>RenduStage</strong> également.<br>
Le programme demandera dans un premier temps de sélectionner un dossier contenant un réseau de neurone (entraîné au préalable via <strong>fichier_train_generique.py</strong>), puis une image à prédire.<br>
Un rapport est disponible dans RenduStage/Rapport_StageLoto.pdf poura voir une vue globabe du stage effectué.
