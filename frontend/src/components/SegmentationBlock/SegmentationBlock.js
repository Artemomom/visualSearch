import React from 'react';
import PropTypes from 'prop-types';
import { makeStyles } from '@material-ui/core/styles';
import Accordion from '@material-ui/core/Accordion';
import AccordionSummary from '@material-ui/core/AccordionSummary';
import AccordionDetails from '@material-ui/core/AccordionDetails';
import Typography from '@material-ui/core/Typography';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import './SegmentationBlock.scss';

const useStyles = makeStyles((theme) => ({
  root: {
    width: '100%',
    boxShadow: 'none'
  },
  heading: {
    color: '#4b4c4d',
    fontSize: theme.typography.pxToRem(15),
    fontWeight: theme.typography.fontWeightRegular
  }
}));

const SegmentationBlock = ({ image, stats }) => {
  const classes = useStyles();

  let detector_stats;
  if ("bbox_detector_time" in stats.perf) {
    detector_stats = `object detection time = ${stats.perf.bbox_detector_time} seconds`
  } else {
    detector_stats = 'object detection is disabled';
  }

  return (
    <div className="segmentation-container">
      <Accordion className={classes.root}>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="panel1a-content"
          id="panel1a-header">
          <Typography className={classes.heading}>
            Show image after object detection
          </Typography>
        </AccordionSummary>
        <AccordionDetails
          style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <img
            src={image}
            alt="targetImg"
            style={{ maxHeight: 200, margin: '0 auto' }}
          />
          <p className="stats">
            {stats.index_size} products in the index <br/>
            {detector_stats},
            vectorization time = {stats.perf.model_time} seconds <br/>
            Model version: {stats.model_version}
          </p>
        </AccordionDetails>
      </Accordion>
    </div>
  );
};

SegmentationBlock.propTypes = {};

export default SegmentationBlock;
