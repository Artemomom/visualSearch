import React from 'react';
import PropTypes from 'prop-types';
import withWidth, { isWidthUp } from '@material-ui/core/withWidth';
import GridList from '@material-ui/core/GridList';
import GridListTile from '@material-ui/core/GridListTile';
import GridListTileBar from '@material-ui/core/GridListTileBar';

import './ImageList.scss';

const ImageList = ({ images, width, image, clearImages }) => {
  const getColsParams = () => {
    if (isWidthUp('xl', width)) {
      return {
        cols: 4,
        height: 450
      };
    }

    if (isWidthUp('lg', width)) {
      return {
        cols: 3,
        height: 450
      };
    }

    if (isWidthUp('md', width)) {
      return {
        cols: 2,
        height: 450
      };
    }

    return {
      cols: 1,
      height: 450
    };
  };

  const deletePreview = () => {
    clearImages();
  };

  return (
    <div className="root">
      <h2 className="list-title"> Most visually similar images </h2>
      <div className="wrapper">
        <div className="thumbWrapper">
          {/*<div className="close-btn" onClick={deletePreview}>
            X
          </div>*/}
          <img src={image} className="preview" alt="preview" />
        </div>
        <div className="grid">
          <GridList
            cols={getColsParams().cols}
            spacing={5}
            cellHeight={getColsParams().height}>
            {images.map((image) => (
              <GridListTile key={image.title}>
                <img
                  src={image.src}
                  alt={image.name}
                  style={{ width: '100%', height: '100%', objectFit: 'scale-down' }}
                />
                <GridListTileBar title={image.name} subtitle={image.subtitle} />
              </GridListTile>
            ))}
          </GridList>
        </div>
      </div>
    </div>
  );
};

ImageList.propTypes = {
  images: PropTypes.array
};

export default withWidth()(ImageList);
