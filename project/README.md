# cs638_Final_Project
Final Project for Building Deep Neural Networks Class (Voice Classification of Audiobooks)

# Data
Inside the raw_data folder, the data are stored in mp3 format.  
If you want to convert it to wav file for easy analyzing, you can use ffmpeg.


```
ffmpeg -i input.mp3 output.wav
```

Some processed data is stored in working_data folder. file_x.wav where x is the duration of the audio file. If x does not exist, it is the original length.  

Breaking wav file into chunks of x seconds with starting time x0 and ending time x0 + x

```
ffmpeg -ss x0 -t x0+x -i input.mp3 output.wav
```

Some References on Audio Processing:  
1) [Wav File Explained](https://blogs.msdn.microsoft.com/dawate/2009/06/22/intro-to-audio-programming-part-1-how-audio-data-is-represented/)  
2) [Wav File Explained 2](https://blogs.msdn.microsoft.com/dawate/2009/06/23/intro-to-audio-programming-part-2-demystifying-the-wav-format/)  

Some libraries to convert wav file or signals into feature vectors (Useful for plotting spectogram and stuff) (Or calculating audio stuff)  
1) [Librosa](https://librosa.github.io/librosa/index.html) (Currently using this <Wayne>)  
2) [Madmom](http://madmom.readthedocs.io/en/latest/index.html)  
3) [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)  
