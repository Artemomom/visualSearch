import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import TextField from '@material-ui/core/TextField';
import CircularProgress from '@material-ui/core/CircularProgress';
import Button from '../Button';
import Dropzone from '../Dropzone/Dropzone';
import { retrieveImageFromClipboardAsBlob } from '../../utils/clipboard';
import './ImageUploadForm.scss';

const ImageUploadForm = ({ images, showImages, clearImages }) => {
  const [url, setUrl] = useState('');
  const [image, setImage] = useState(null);
  const [showLoader, setShowLoader] = useState(false);

  const onUrlChange = (e) => {
    const value = e.target.value;
    setUrl(value);
  };

  const findSimilarImages = async () => {
    clearImages();
    setShowLoader(true);
    if (image) {
      const formData = new FormData();
      formData.append('image', image);
      formData.append('filename', image.name);
      await showImages(formData);
    } else if (url) {
      const formData = new FormData();
      formData.append('url', url);
      await showImages(formData);
    }
    setShowLoader(false);
  };

  const handleClipboard = (e) => {
    retrieveImageFromClipboardAsBlob(e, (blob) => {
      if (blob) {
        const image = new File([blob], 'name');

        // Crossbrowser support for URL
        const URLObj = window.URL || window.webkitURL;
        image.preview = URLObj.createObjectURL(blob);
        setImage(image);
      }
    });
  };

  useEffect(() => {
    window.addEventListener('paste', handleClipboard, false);
    return () => {
      window.removeEventListener('keydown', handleClipboard);
    };
  }, []);

  useEffect(() => {
    if (image) {
      findSimilarImages();
    }
  }, [image]);

  // const deletePreview = () => {
  //   setImage(null);
  //   setUrl('');
  //   clearImages();
  // };

  return (
    <div
      className="image-upload-form"
      style={{
        marginTop: images?.length ? '50px' : 0
      }}>
      <p className="title">Please select image with product you are looking for</p>
      <div className="button-wrapper">
        <Dropzone chooseImage={setImage} />
      </div>
      <p className="or">OR</p>
      <TextField
        className="text-input"
        id="standard-basic"
        label="URL"
        margin="dense"
        value={url}
        onChange={onUrlChange}
      />
      {/* <div className="thumbContainer">
        {(image?.preview || url) && (
          <div className="thumb" key={image?.name}>
            <div className="thumbInner">
              <img
                src={image?.preview || url}
                className="preview-image"
                alt="preview"
              />
            </div>
            <div className="close-button" onClick={deletePreview}>
              X
            </div>
          </div>
        )}
      </div> */}
      <div className="button-wrapper">
        <Button
          title="Find similar"
          onClick={findSimilarImages}
          disabled={showLoader || (url === '' && image == null)}
          padding="10px 30px"
        />
      </div>
      <div className="loader-wrapper">
        {showLoader && <CircularProgress color="primary" />}
      </div>
    </div>
  );
};

ImageUploadForm.propTypes = {
  images: PropTypes.array
};

export default ImageUploadForm;
