import React, { useEffect, useState } from 'react';
import { Alert } from '@material-ui/lab';
import { withLayout } from '../hoc/withLayout';
import ImageUploadForm from '../components/ImageUploadForm/ImageUploadForm';
import ImageList from '../components/ImageList';
import Api from '../api/api';
import './home.scss';
import SegmentationBlock from '../components/SegmentationBlock/SegmentationBlock';

const api = new Api();

const Home = () => {
  const [images, setImages] = useState(null);
  const [targetImg, setTargetImg] = useState(null);
  const [segmentedImg, setSegmentedImg] = useState(null);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    if (error) {
      setTimeout(() => {
        setError(null);
      }, 15000);
    }
  }, [error]);

  const showImages = async (data) => {
    try {
      const response = await api.getSimilarImages(data);
      const similarPaths = response.similar_paths.map((path, index) => ({
        name: path[2],
        subtitle: path[1],
        title: `${index + 1}`,
        src: path[0]
      }));
      setImages(similarPaths);
      setTargetImg(response.target_img);
      setSegmentedImg(response.img_after_segmentation);
      setStats(response.stats);
    } catch (e) {
      setError(e.message);
      console.error(e);
    }
  };

  const clearImages = () => {
    setImages(null);
    setTargetImg(null);
    setSegmentedImg(null);
    setStats(null);
  };

  return (
    <div
      className="container"
      style={{
        justifyContent: images ? 'unset' : 'center',
        marginTop: images ? '80px' : 0
      }}>
      <ImageUploadForm
        images={images}
        showImages={showImages}
        clearImages={clearImages}
      />
      {segmentedImg && stats && <SegmentationBlock image={segmentedImg} stats={stats}/>}
      {images && stats && targetImg && (
        <ImageList
          images={images}
          stats={stats}
          image={targetImg}
          clearImages={clearImages}
        />
      )}
      <div className={`error-message${error ? '' : ' hide'}`}>
        <Alert severity="error" color="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </div>
    </div>
  );
};

export default withLayout(Home);
