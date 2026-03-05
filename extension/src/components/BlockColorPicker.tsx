import React, { useState, useEffect } from "react";
import { OverlayTrigger, Popover, Button } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPalette } from "@fortawesome/free-solid-svg-icons";
import { useBlockColor } from "../context/BlockColorContext";
import "./BlockColorPicker.css";

function BlockColorPicker() {
  const { blockColor, setBlockColor } = useBlockColor();
  const [hexInput, setHexInput] = useState(blockColor);

  useEffect(() => {
    setHexInput(blockColor);
  }, [blockColor]);

  const handleHexChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value;
    setHexInput(v);
    if (/^#[0-9A-Fa-f]{6}$/.test(v)) {
      setBlockColor(v);
    } else if (/^#[0-9A-Fa-f]{3}$/.test(v)) {
      const expanded = v.replace(/#(\w)(\w)(\w)/, "#$1$1$2$2$3$3");
      setBlockColor(expanded);
    }
  };

  const popover = (
    <Popover className="block-color-popover" id="block-color-popover">
      <Popover.Header as="h3">Block color</Popover.Header>
      <Popover.Body>
        <div className="block-color-picker-row">
          <input
            type="color"
            value={blockColor}
            onChange={(e) => setBlockColor(e.target.value)}
            className="block-color-input"
            aria-label="Choose block color"
          />
          <input
            type="text"
            value={hexInput}
            onChange={handleHexChange}
            className="block-color-hex"
            placeholder="#212529"
            maxLength={7}
            aria-label="Hex color"
          />
        </div>
      </Popover.Body>
    </Popover>
  );

  return (
    <div className="block-color-picker-top-left">
      <OverlayTrigger
        trigger="click"
        placement="bottom-start"
        overlay={popover}
        rootClose
      >
        <Button
          variant="link"
          className="block-color-palette-btn"
          aria-label="Open block color picker"
        >
          <FontAwesomeIcon icon={faPalette} className="block-color-palette-icon" />
        </Button>
      </OverlayTrigger>
    </div>
  );
}

export default BlockColorPicker;
