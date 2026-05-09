from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from typing import Annotated, List, Literal, Optional, Union
from dateutil import parser
import re

# --- Define Helper Logic for Validation ---
# Function to clean and convert currency strings to float
def clean_currency_to_format(v: str) -> float:
    """Removes currency symbols, spaces, and commas, then returns a float."""
    if not v: return 0.0
    # Remove everything except numbers and the decimal point
    cleaned = re.sub(r'[^\d.]', '', v.replace(',', ''))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

# Function to clean and standardize date formats
def clean_date_format(v: str) -> str:
    if not v or v.lower() in ["n/a", "none", ""]:
        return None
    try:
        # Remove common separators
        v_cleaned = re.sub(r"[,.]", " ", v).strip()

        # Parse intelligently. 
        dt = parser.parse(v_cleaned, dayfirst=True, fuzzy=True)
        
        # Return your specific format
        return dt.strftime("%d-%m-%Y")
    except (ValueError, TypeError):
        return v

# --- Define Pydantic Models ---
# These models define the expected structure of the extracted data 
class InvoiceItem(BaseModel):
    # Using String for 'no' because line numbers can be "1", "1a", "A", etc.
    no: Optional[str] = Field(None, validation_alias=AliasChoices("no", "item_no"), description="The line item number or identifier")

    desc: Optional[str] = Field(None, validation_alias=AliasChoices("desc", "description"), description="Description of the product or service")

    # Quantity is usually safe as a float, but Optional handles missing data
    quantity: Optional[float] = Field(None, validation_alias=AliasChoices("quantity", "qty"), description=" The count or amount of items")

    unit: Optional[str] = Field(None, validation_alias=AliasChoices("unit", "uom"), description="Unit of measurement (e.g., kg, hours, pcs)")

    # Keeping money as strings to prevent parsing errors on currency symbols (e.g. "$1,200.00")
    net_price: Optional[float] = Field(None, validation_alias=AliasChoices("net_price", "unit_price"), description="Price per unit")
    net_worth: Optional[float] = Field(None, validation_alias=AliasChoices("net_worth", "subtotal"), description="Total net value before tax")
    
    # Renamed from "VAT_%" to valid Python syntax
    vat_percentage: Optional[float] = Field(None, validation_alias=AliasChoices("vat_percentage", "tax_rate", "vat_rate"), description="VAT/Tax percentage applied")

    gross_worth: Optional[float] = Field(None, validation_alias=AliasChoices("gross_worth", "total_price"),description="Total value including tax")

    @field_validator("quantity", "net_price", "net_worth", "gross_worth", "vat_percentage", mode="before")
    # @field_validator("net_price", "net_worth", "gross_worth", mode="before")
    @classmethod
    def validate_price(cls, v):
        if isinstance(v, str):
            return clean_currency_to_format(v)
        return v
    
class InvoiceTotal(BaseModel):
    net_total: Optional[float] = Field(None, validation_alias=AliasChoices("net_total", "total_net"), description="Total net amount before tax")
    total_vat: Optional[float] = Field(None, validation_alias=AliasChoices("total_vat", "vat_amount", "tax_amount"), description="Total VAT/Tax amount")
    gross_total: Optional[float] = Field(None, validation_alias=AliasChoices("gross_total", "total_amount", "amount_due"), description="Total amount including tax")

    @field_validator("net_total", "total_vat", "gross_total", mode="before")
    @classmethod
    def validate_price(cls, v):
        if isinstance(v, str):
            return clean_currency_to_format(v)
        return v

class InvoiceHeader(BaseModel):

    invoice_number: Optional[str] = Field(None, validation_alias=AliasChoices("invoice_number", "inv_no", "bill_id", "reference"), description="Unique alphanumeric identifier of the invoice")
    date_of_issue: Optional[str] = Field(None, validation_alias=AliasChoices("date_of_issue", "date", "billing_date"),description="The date the invoice was issued in DD-MM-YYYY format")
    
    seller_name: Optional[str] = Field(None, validation_alias=AliasChoices("seller_name", "vendor", "supplier", "from"), description="Name of the vendor or supplier")
    seller_address: Optional[str] = Field(None, validation_alias=AliasChoices("seller_address", "vendor_address", "supplier_address", "address_from"), description="Full address of the seller")

    buyer_name: Optional[str] = Field(None, validation_alias=AliasChoices("buyer_name", "customer", "bill_to", "client"), description="Name of the client or customer")
    buyer_address: Optional[str] = Field(None, validation_alias=AliasChoices("buyer_address", "customer_address", "bill_to_address", "address_to"), description="Full address of the buyer")

    @field_validator("date_of_issue", mode="before")
    @classmethod
    def format_date(cls, v):
        if isinstance(v, str):
            return clean_date_format(v)
        return v

class InvoiceAll(BaseModel):
    # This Literal is the 'Discriminator'
    document_type: Literal["invoice"] = "invoice"
    
    header: InvoiceHeader = Field(..., description="General information about the invoice, seller, and buyer")
    items: List[InvoiceItem] = Field(default_factory=list, description="List of individual line items or products")
    totals: InvoiceTotal = Field(..., description="Summary of the financial totals including tax")

# --- The Extensible Wrapper ---
# Add new document types to this Union list as you grow
DocumentUnion = Annotated[
    Union[InvoiceAll], # add more document classes here
    Field(discriminator="document_type")
]

class UniversalExtraction(BaseModel):
    # This allows the model to accept either 'document_type' or its aliases
    model_config = ConfigDict(populate_by_name=True)
    data: DocumentUnion

    def get_searchable_payload(self) -> dict:
        """
        Returns flat fields for Qdrant filtering 
        AND the full data for the Agent to read.
        """
        if isinstance(self.data, InvoiceAll):
            return {
                # 1. FLAT FIELDS (For Qdrant Metadata Filtering)
                "invoice_number": self.data.header.invoice_number,
                "seller_name": self.data.header.seller_name,
                "gross_total": self.data.totals.gross_total,
                "date": self.data.header.date_of_issue,
                
                # 2. THE BLOB (For the Pydantic AI Agent to read)
                # We store the full nested dict (including items) here.
                "full_extraction": self.data.model_dump() 
            }
        return {}